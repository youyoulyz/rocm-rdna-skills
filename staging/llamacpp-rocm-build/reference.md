# llamacpp-rocm-build — Reference

Deep reference for building llama.cpp with ROCm on RDNA consumer GPUs.

## Build Options

| Option | RDNA 3 (gfx1100) | RDNA 2 (gfx1030) | Notes |
|--------|------------------|------------------|-------|
| `GGML_HIP=ON` | yes | yes | The HIP backend. Without it cmake silently falls back to CPU-only. |
| `GGML_HIP_ROCWMMA_FATTN` | OFF (see below) | OFF (required) | RDNA2 has no WMMA at all. On RDNA3, rocWMMA flash-attn is often slower/less stable than the plain path; this repo's verified builds use OFF. |
| `AMDGPU_TARGETS` | `gfx1100` | `gfx1030` | Single target = faster compile (~5 min). All targets = ~40 min. |
| `CMAKE_BUILD_TYPE` | Release | Release | Required for any usable performance. |
| `GGML_HIP_MMQ_MFMA` | default | default | MFMA kernels are CDNA-optimized; on RDNA the MMQ path is used anyway. |
| `GGML_HIP_NO_VMM` | default | default | VMM (virtual memory mapping) is unsupported on consumer RDNA — off by default in recent builds. |

### ROCWMMA on RDNA 3 — why OFF

rocWMMA flash attention was enabled by default in some llama.cpp versions
for HIP builds. On RDNA 3 it can be slower than the non-WMMA attention
path and occasionally crashes with `ROCWMMA` kernel errors. The verified
builds on this repo's target machine all use `GGML_HIP_ROCWMMA_FATTN=OFF`.
If you want to experiment: `-DGGML_HIP_ROCWMMA_FATTN=ON` and compare
`llama-bench` output. Flash attention via rocWMMA does help on models
with large contexts and many layers — benchmark before deciding.

## Environment

The cmake HIP path requires:

```bash
HIPCXX="$(hipconfig -l)/clang"
HIP_PATH="$(hipconfig -R)"
```

- `hipconfig` must be on PATH (ROCm install adds /opt/rocm/bin).
- Missing `libhipblas-dev` / `librocblas-dev` makes the HIP backend
  silently unavailable — the build completes but every layer runs on
  CPU. Always check the cmake output for "HIP" in the backend list, or
  verify with `scripts/verify.py` which greps for the GPU device line.

### Stale HIP compiler in CMakeCache after a ROCm upgrade

`CMakeCache.txt` pins `CMAKE_HIP_COMPILER` to the absolute compiler path
from the ROCm version the build was first configured with. After a ROCm
upgrade (e.g. 7.2.0 → 7.2.4) that path is stale, and cmake reuses the
cached value on reconfigure — **ignoring the HIPCXX env var** — failing
with:

```
The CMAKE_HIP_COMPILER: /opt/rocm-7.2.0/lib/llvm/bin/clang++
is not a full path to an existing compiler tool.
```

Fix (what `build.py` does automatically): pass
`-DCMAKE_HIP_COMPILER=$(hipconfig -l)/clang++` on the configure line to
override the stale cache entry. If that still fails, the cache can be
cleared for HIP variables only:

```bash
cmake -S . -B build_base -U 'CMAKE_HIP*'
```

## Typical Build Times

| Setup | Time |
|-------|------|
| gfx1100 only, 24 cores, Release | ~4-6 min |
| All gfx targets | 30-45 min |
| Incremental rebuild after git pull | ~1-3 min |

## Updating

- `git pull --ff-only origin master` — llama.cpp uses `master`.
- Uncommitted local edits block the pull (`git stash` first, or the
  build script reports the conflict and stops).
- Check `AGENTS.md` at the repo root after updates — llama.cpp documents
  build conventions there.

## Verification

`scripts/verify.py` checks in order:

1. `llama-server` exists in the build dir.
2. `llama-server --version` runs.
3. The server output lists the GPU device line
   (`Device 0: AMD Radeon RX 7900 XTX, gfx1100 (0x1100)`).
4. (optional, with `--model`) A real model loads with `--n-gpu-layers 99`
   and generates a completion.

A GPU device line with `gfx1100` proves the HIP backend linked and the
HSA runtime found the card. If step 3 fails but the binary exists, the
most likely cause is a HIP-less build (see Environment above).

## Multiple Build Directories

It is common to keep several build dirs (e.g. `build_base` for the
standard build, `build-hip-moe-*` for experiments). The skill's
`build.py` picks `build_base` by default — the dir whose CMakeCache has
`AMDGPU_TARGETS` set — and inherits its options, so rebuilding after a
pull produces the same configuration you had, not a generic one.
