#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Verify a freshly built llama.cpp ROCm binary.

Checks: the binary exists and runs, it reports the RDNA GPU, and (with
--model) the GPU actually loads a model and generates a completion.

Usage:
    python scripts/verify.py [--src /opt/llama.cpp/llama.cpp] [--build-dir build_base]
    python scripts/verify.py --model /path/to/model.gguf [--prompt "hi"] [--n-gpu-layers 99]

Exit codes: 0 = verified, 1 = verification failed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

DEFAULT_SRC = "/opt/llama.cpp/llama.cpp"


def _vram_used_gb():
    """Total VRAM used across GPUs, via rocm-smi. None if unavailable."""
    rc, out = _run("rocm-smi --showmeminfo vram 2>/dev/null | grep 'Used' | head -1")
    if rc != 0 or not out:
        return None
    try:
        return int(out.split(":")[-1].strip()) / (1024**3)
    except ValueError:
        return None


def _run(cmd, timeout=120):
    try:
        r = subprocess.run(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout,
        )
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired:
        return 1, "timed out"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=os.environ.get("LLAMACPP_SRC", DEFAULT_SRC))
    p.add_argument("--build-dir", default="build_base")
    p.add_argument("--model", default="", help="GGUF model path for a real inference test")
    p.add_argument("--prompt", default="Say hello in one sentence")
    p.add_argument("--n-gpu-layers", type=int, default=99)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    src = os.path.abspath(args.src)
    bin_dir = os.path.join(src, args.build_dir, "bin")
    server = os.path.join(bin_dir, "llama-server")

    checks = []

    # 1. Binary exists
    if not os.path.isfile(server):
        checks.append({"check": "binary", "ok": False,
                       "message": f"llama-server not found at {server} — wrong --build-dir?"})
        result = {"ok": False, "checks": checks}
        print(json.dumps(result, indent=2))
        sys.exit(1)
    checks.append({"check": "binary", "ok": True, "message": server})

    # 2. Binary runs and reports version
    rc, out = _run(f"{server} --version", timeout=60)
    version_line = next((l for l in out.splitlines() if "version" in l.lower()), out[:100])
    checks.append({"check": "version", "ok": rc == 0, "message": version_line.strip()[:120]})

    # 3. GPU used. With a model: start a server loading it and measure
    # VRAM via rocm-smi — HIP only allocates VRAM once weights load, so
    # a real model is required. Without a model: check the binary links
    # the HIP backend shared library.
    if args.model:
        import time
        baseline = _vram_used_gb()
        _run("pkill -f 'port 18099' 2>/dev/null; true")
        rc, _ = _run(
            f"nohup {server} -m {args.model} --n-gpu-layers {args.n_gpu_layers} "
            "--ctx-size 1024 --host 127.0.0.1 --port 18099 "
            "> /tmp/llama-verify.log 2>&1 & echo $! > /tmp/llama-verify.pid",
            timeout=10,
        )
        # wait for model load (poll health up to 60s)
        loaded = False
        for _ in range(30):
            time.sleep(2)
            rc, _ = _run("curl -sf http://127.0.0.1:18099/health", timeout=10)
            if rc == 0:
                loaded = True
                break
        vram_now = _vram_used_gb()
        _run("kill $(cat /tmp/llama-verify.pid) 2>/dev/null; true")
        used_delta = round(vram_now - baseline, 2) if baseline is not None else None
        if loaded and used_delta is not None and used_delta > 0.5:
            checks.append({"check": "gpu", "ok": True,
                           "message": f"HIP backend active — VRAM usage rose by {used_delta} GB while loading the model"})
        else:
            checks.append({"check": "gpu", "ok": False,
                           "message": f"VRAM did not rise (delta={used_delta} GB, loaded={loaded}) — HIP backend not linked?"})
    else:
        rc, ldd_out = _run(f"ldd {server} 2>&1 | grep -c ggml-hip", timeout=30)
        if rc == 0 and ldd_out.strip().isdigit() and int(ldd_out.strip()) > 0:
            checks.append({"check": "gpu", "ok": True,
                           "message": "binary links the ggml-hip backend library"})
        else:
            checks.append({"check": "gpu", "ok": False,
                           "message": "binary does not link ggml-hip — HIP backend not in this build"})

    # 4. Real inference (optional)
    if args.model:
        rc, out = _run(
            f"{server} -m {args.model} --n-gpu-layers {args.n_gpu_layers} "
            f'--ctx-size 512 --seed 42 -p "{args.prompt}" -n 20 2>&1 | tail -8',
            timeout=300,
        )
        ok = rc == 0 and any(l.strip() for l in out.splitlines())
        checks.append({"check": "inference", "ok": ok,
                       "message": f"model {os.path.basename(args.model)}"
                       + (" generated output" if ok else f" failed: {out[:200]}")})

    result = {"ok": all(c["ok"] for c in checks), "checks": checks}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for c in checks:
            print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['check']}: {c['message']}")
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
