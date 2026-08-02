#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Detect the state of a llama.cpp ROCm source build.

Read-only. Answers: where is the llama.cpp source, is it a git checkout,
what commit is it on, how far behind origin, what GPU/ROCm is present,
and what were the previous cmake build options.

Usage:
    python scripts/detect.py [--src /opt/llama.cpp/llama.cpp] [--json]

Exit codes: 0 = buildable checkout found, 1 = source missing, 2 = not a
git checkout, 3 = no RDNA GPU / ROCm (nothing to build for).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

DEFAULT_SRC = "/opt/llama.cpp/llama.cpp"


def _run(cmd, cwd=None, timeout=30):
    try:
        r = subprocess.run(
            cmd, shell=True, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout,
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", f"Command timed out after {timeout}s"


def _gfx_target():
    """Detect the primary RDNA gfx target via amd-smi or fallback."""
    rc, out, err = _run("amd-smi static --asic --json")
    if rc != 0 and "required groups" in err:
        rc, out, _ = _run("sudo amd-smi static --asic --json")
    if rc == 0:
        try:
            import json as j
            data = j.loads(out)
            entries = data if isinstance(data, list) else data.get("gpu_data", [data])
            for e in entries:
                gfx = e.get("asic", {}).get("target_graphics_version", "")
                if gfx:
                    return gfx.lower()
        except (json.JSONDecodeError, AttributeError):
            pass
    # Fallback: rocminfo
    rc, out, _ = _run("rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1")
    if rc == 0 and out:
        return out
    return None


def _rocm_version():
    rc, out, _ = _run("cat /opt/rocm/.info/version 2>/dev/null")
    return out if rc == 0 else None


def _existing_builds(src):
    """List build directories and their cmake options."""
    builds = []
    if not os.path.isdir(src):
        return builds
    for entry in sorted(os.listdir(src)):
        if not entry.startswith("build"):
            continue
        cache = os.path.join(src, entry, "CMakeCache.txt")
        if not os.path.isfile(cache):
            continue
        opts = {}
        for var in ("GGML_HIP", "GGML_HIP_ROCWMMA_FATTN",
                    "AMDGPU_TARGETS", "CMAKE_BUILD_TYPE"):
            rc, out, _ = _run(f"grep -E '^{var}:' {cache}")
            if rc == 0 and out:
                opts[var] = out.split("=", 1)[-1]
        builds.append({"dir": entry, "cmake": opts})
    return builds


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=os.environ.get("LLAMACPP_SRC", DEFAULT_SRC))
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    src = os.path.abspath(args.src)

    # Source directory
    if not os.path.isdir(src):
        print(json.dumps({"error": f"llama.cpp source not found at {src}",
                          "hint": "git clone https://github.com/ggml-org/llama.cpp.git <src>"}))
        sys.exit(1)

    # Git checkout?
    rc, remote, _ = _run("git remote get-url origin", cwd=src)
    if rc != 0:
        print(json.dumps({"error": f"{src} is not a git checkout of llama.cpp"}))
        sys.exit(2)

    rc, cur_commit, _ = _run("git rev-parse --short HEAD", cwd=src)
    rc2, cur_subj, _ = _run("git log --oneline -1", cwd=src)

    # Behind origin?
    behind = 0
    ahead = 0
    rc3, _, _ = _run("git fetch origin --quiet", cwd=src, timeout=60)
    if rc3 == 0:
        rc4, behind_out, _ = _run("git rev-list --count HEAD..origin/master", cwd=src)
        rc5, ahead_out, _ = _run("git rev-list --count origin/master..HEAD", cwd=src)
        behind = int(behind_out) if rc4 == 0 and behind_out.isdigit() else 0
        ahead = int(ahead_out) if rc5 == 0 and ahead_out.isdigit() else 0

    # Working tree dirty?
    rc6, dirty, _ = _run("git status --porcelain | wc -l", cwd=src)
    dirty_count = int(dirty) if rc6 == 0 and dirty.isdigit() else -1

    result = {
        "src": src,
        "remote": remote,
        "current_commit": cur_commit,
        "current_subject": cur_subj,
        "commits_behind_origin": behind,
        "commits_ahead_origin": ahead,
        "dirty_files": dirty_count,
        "gfx_target": _gfx_target(),
        "rocm_version": _rocm_version(),
        "existing_builds": _existing_builds(src),
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Source:     {src}")
        print(f"Remote:     {remote}")
        print(f"Commit:     {cur_commit}  {cur_subj}")
        print(f"Behind:     {behind} commits behind origin/master"
              + (f", {ahead} ahead" if ahead else ""))
        print(f"Dirty:      {dirty_count} uncommitted file(s)")
        print(f"GPU:        {result['gfx_target'] or 'unknown'}")
        print(f"ROCm:       {result['rocm_version'] or 'unknown'}")
        for b in result["existing_builds"]:
            opts = " ".join(f"{k}={v}" for k, v in b["cmake"].items())
            print(f"Build:      {b['dir']}  [{opts}]")

    # Exit codes: 3 = no GPU/ROCm
    if not result["gfx_target"] or not result["rocm_version"]:
        sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    main()
