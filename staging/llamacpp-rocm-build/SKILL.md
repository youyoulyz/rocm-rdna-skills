---
name: llamacpp-rocm-build
description: >-
  Updates and compiles llama.cpp from source with ROCm/HIP support for AMD RDNA
  GPUs. Use when the user wants to build, compile, rebuild, update, or upgrade
  llama.cpp with GPU support on an AMD consumer GPU (RX 7900 XTX, RX 7000/6000
  series, gfx1100). Handles: detecting the source checkout state, pulling the
  latest upstream commits, configuring cmake with the HIP backend for the
  detected gfx target, compiling, and verifying the binary sees the GPU and
  generates output. Also use when the user mentions "build llama.cpp with
  ROCm", "llama.cpp HIP build", or "recompile llama.cpp". Do not use for
  NVIDIA GPUs (CUDA build) or for serving models (use serving-llms-on-rdna).
---

# Build llama.cpp with ROCm on RDNA

Pull the latest llama.cpp and compile it with the HIP (ROCm) backend for
the machine's RDNA GPU. The flow preserves your existing build
configuration — it inherits the cmake options from the previous build
dir instead of inventing new ones.

## Prerequisites

- llama.cpp source checkout (default `/opt/llama.cpp/llama.cpp`)
- ROCm with `hipcc`, `hipconfig`, and `libhipblas-dev` installed
- RDNA GPU detected (gfx1100 for RX 7900 XTX)

## Flow

### Step 1: Detect the current state

```bash
uv run --quiet scripts/detect.py --json
```

This reports:
- Source path, git remote, current commit
- **commits_behind_origin** — how stale the checkout is
- **dirty_files** — uncommitted local changes (they can block the pull)
- **gfx_target** and **rocm_version**
- **existing_builds** — every build dir with its inherited cmake options

If the source is missing (exit 1): clone it —
`git clone https://github.com/ggml-org/llama.cpp.git <src>` — then proceed.
If no RDNA GPU or ROCm (exit 3): report and stop.

### Step 2: Present the plan

Summarize for the user:
- Current commit → how many commits behind origin/master
- Which build dir will be rebuilt (default `build_base` if it exists)
- The cmake options that will be inherited (HIP=ON, ROCWMMA off on RDNA3)
- Expected compile time (~5 min for gfx1100-only)

If `dirty_files > 0`, warn that the pull may conflict. Ask the user
whether to keep local changes (stash) or leave them.

### Step 3: Pull and build

```bash
uv run --quiet scripts/build.py --json
```

The script:
1. `git pull --ff-only origin master` (falls back to `main`)
2. Configures cmake with the inherited options:
   `-DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=<inherited> -DAMDGPU_TARGETS=<gfx> -DCMAKE_BUILD_TYPE=Release`
3. Compiles with `-j$(nproc)`
4. Prints the built binaries and the resulting commit

Override the build dir with `--build-dir <name>` (e.g. to rebuild
`build-hip-moe-rpb8` instead), or skip the pull with `--no-pull`.

### Step 4: Verify

```bash
uv run --quiet scripts/verify.py --json
```

Checks: binary exists → `--version` runs → server output lists the GPU
device line (`Device 0: AMD Radeon RX 7900 XTX, gfx1100`). The GPU line
proves the HIP backend linked and found the card.

For a real inference test, pass a GGUF model:

```bash
uv run --quiet scripts/verify.py --model /mnt/public/GGUF_models/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

### Step 5: Report

Present:
- New commit hash + subject
- Build dir, binaries path (`<src>/<build_dir>/bin/`)
- Verification results (version, GPU device line, inference if tested)
- How to serve: `llama-server -m <model.gguf> --n-gpu-layers 99 --host 0.0.0.0 --port 8080`
  (or hand off to `serving-llms-on-rdna`)

## Safety Rules

1. Never delete or overwrite a user's experimental build dir
   (`build-hip-*` variants) — only `build_base`/`build` are touched by
   default, and only via cmake reconfigure + build.
2. `git pull` is `--ff-only`: if local commits conflict, stop and ask.
3. Uncommitted file changes are reported, never discarded.
4. The build inherits the previous cmake options — do not silently flip
   `GGML_HIP_ROCWMMA_FATTN` or other flags without telling the user.
5. Compiling is the only heavy operation; a failed build leaves the old
   binaries in place until cmake overwrites them.

## Verification Checklist

- [ ] `detect.py` reports the source, commit, and how far behind origin
- [ ] User approved the rebuild plan (build dir + options)
- [ ] `build.py` exits 0 after pulling and compiling
- [ ] `verify.py` passes: binary runs, GPU device line present
- [ ] (optional) inference test with a real GGUF model succeeds
