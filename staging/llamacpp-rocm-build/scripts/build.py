#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Update and rebuild llama.cpp with ROCm/HIP support.

Pulls the latest origin/master into the source checkout, configures
cmake with the same options the previous build used (inherited from the
existing build dir's CMakeCache), and compiles.

Build options are inherited, not invented: if the previous build had
GGML_HIP_ROCWMMA_FATTN=OFF (common on RDNA3), the rebuild keeps it OFF.
Only AMDGPU_TARGETS is forced if it was missing, to avoid building all
gfx targets.

Usage:
    python scripts/build.py [--src /opt/llama.cpp/llama.cpp] [--build-dir build_base] [--jobs 24] [--no-pull]

Exit codes: 0 = build succeeded, 1 = pull/configure/build failed.

Env vars:
    LLAMACPP_SRC  -- source checkout (default /opt/llama.cpp/llama.cpp)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

DEFAULT_SRC = "/opt/llama.cpp/llama.cpp"


def _run(cmd, cwd=None, timeout=600, check=False):
    try:
        r = subprocess.run(
            cmd, shell=True, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout,
        )
        if check and r.returncode != 0:
            raise BuildError(f"command failed ({r.returncode}): {cmd}\n{r.stdout[-2000:]}")
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired as e:
        raise BuildError(f"command timed out: {cmd}") from e


class BuildError(Exception):
    pass


def _pick_build_dir(src, requested):
    """Choose which existing build dir to reuse.

    Preference: explicit --build-dir > build_base (standard build with
    AMDGPU_TARGETS set) > build > any build-* dir.
    """
    if requested:
        d = os.path.join(src, requested)
        if not os.path.isdir(d):
            raise BuildError(f"requested build dir {requested} does not exist")
        return requested
    for name in ("build_base", "build"):
        if os.path.isdir(os.path.join(src, name)):
            return name
    entries = sorted(os.listdir(src))
    for entry in entries:
        if entry.startswith("build-") and os.path.isdir(os.path.join(src, entry)):
            return entry
    return "build"


def _inherit_cmake_options(src, build_dir):
    """Read cmake options from the previous build's CMakeCache."""
    cache = os.path.join(src, build_dir, "CMakeCache.txt")
    opts = {}
    if os.path.isfile(cache):
        rc, out = _run(f"grep -E '^(GGML_HIP|GGML_HIP_ROCWMMA_FATTN|CMAKE_BUILD_TYPE):' {cache}")
        for line in out.splitlines():
            if "=" in line:
                var, val = line.split("=", 1)
                var = var.split(":")[0]
                opts[var] = val
    return opts


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=os.environ.get("LLAMACPP_SRC", DEFAULT_SRC))
    p.add_argument("--build-dir", default="", help="cmake build dir under src (default: auto-pick)")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 8)
    p.add_argument("--no-pull", action="store_true", help="Skip git pull (rebuild only)")
    p.add_argument("--gfx", default="", help="AMDGPU_TARGETS override (default: detect or inherit)")
    p.add_argument("--json", action="store_true", help="Emit only the final JSON result")
    args = p.parse_args()

    src = os.path.abspath(args.src)
    if not os.path.isdir(src):
        print(json.dumps({"error": f"source not found: {src}"}))
        sys.exit(1)

    try:
        # 1. Pull latest
        if not args.no_pull:
            print("== Pulling latest origin/master ==")
            rc, out = _run("git pull --ff-only origin master", cwd=src, timeout=300)
            print(out[-1500:])
            if rc != 0:
                # Try main branch as fallback
                rc, out = _run("git pull --ff-only origin main", cwd=src, timeout=300)
                print(out[-1500:])
                if rc != 0:
                    raise BuildError("git pull failed (local changes conflict?)")

        # 2. Current commit
        rc, commit = _run("git rev-parse --short HEAD", cwd=src)
        print(f"\n== Building at {commit.strip()} ==")

        # 3. Choose build dir and inherit options
        build_dir = _pick_build_dir(src, args.build_dir)
        print(f"== Build dir: {build_dir} ==")
        opts = _inherit_cmake_options(src, build_dir)
        print(f"== Inherited cmake options: {opts or '(none — fresh configure)'} ==")

        hip_flag = opts.get("GGML_HIP", "ON")
        wmma_flag = opts.get("GGML_HIP_ROCWMMA_FATTN", "OFF")
        build_type = opts.get("CMAKE_BUILD_TYPE", "Release")
        gfx = args.gfx or "gfx1100"

        # 4. Configure
        # Explicitly override CMAKE_HIP_COMPILER: an existing CMakeCache
        # pins the compiler path from the ROCm version it was configured
        # with. After a ROCm upgrade (e.g. 7.2.0 -> 7.2.4) the cached
        # /opt/rocm-7.2.0/... path is stale and cmake reuses it, ignoring
        # the HIPCXX env var. Passing -DCMAKE_HIP_COMPILER forces the
        # current toolchain.
        rc_hc, hipcxx_path = _run("hipconfig -l 2>/dev/null", timeout=10)
        if rc_hc != 0 or not hipcxx_path:
            raise BuildError("hipconfig -l failed — is ROCm installed and on PATH?")
        hipcxx = os.path.join(hipcxx_path.strip(), "clang++")
        if not os.path.isfile(hipcxx):
            raise BuildError(f"ROCm clang++ not found at {hipcxx}")
        hippath = "$(hipconfig -R)"
        configure_cmd = (
            f'cd {src} && HIPCXX="{hipcxx}" HIP_PATH="{hippath}" '
            f'cmake -S . -B {build_dir} '
            f"-DCMAKE_HIP_COMPILER={hipcxx} "
            f"-DGGML_HIP={hip_flag} "
            f"-DGGML_HIP_ROCWMMA_FATTN={wmma_flag} "
            f"-DAMDGPU_TARGETS={gfx} "
            f"-DCMAKE_BUILD_TYPE={build_type} "
            "2>&1 | tail -15"
        )
        print(f"== Configuring: -DGGML_HIP={hip_flag} -DGGML_HIP_ROCWMMA_FATTN={wmma_flag} -DAMDGPU_TARGETS={gfx}")
        rc, out = _run(configure_cmd, timeout=300)
        print(out)
        if rc != 0:
            raise BuildError("cmake configure failed")

        # 5. Build
        print(f"\n== Compiling with -j{args.jobs} (this takes several minutes) ==")
        rc, out = _run(
            f"cmake --build {os.path.join(src, build_dir)} --config {build_type} -j{args.jobs} 2>&1 | tail -25",
            timeout=3600,
        )
        print(out)
        if rc != 0:
            raise BuildError("cmake build failed")

        # 6. Verify artifacts
        binaries = ["llama-server", "llama-cli", "llama-quantize"]
        found = [b for b in binaries
                 if os.path.isfile(os.path.join(src, build_dir, "bin", b))]
        print(f"\n== Built: {', '.join(found) or 'no expected binaries found'} ==")

        result = {
            "status": "ok",
            "src": src,
            "build_dir": build_dir,
            "commit": commit.strip(),
            "binaries": found,
            "binary_path": os.path.join(src, build_dir, "bin"),
            "cmake": {
                "GGML_HIP": hip_flag,
                "GGML_HIP_ROCWMMA_FATTN": wmma_flag,
                "AMDGPU_TARGETS": gfx,
                "CMAKE_BUILD_TYPE": build_type,
            },
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result))
        sys.exit(0)

    except BuildError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
