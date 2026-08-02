---
name: install-rocm-rdna-stack
description: >-
  Installs the full vLLM serving stack on AMD RDNA consumer GPUs (RX 7900 XTX /
  RX 7900 XT / gfx1100) from wheels: creates a Python 3.14 venv, installs the
  TheRock pip-built ROCm 7.14 SDK (rocm-sdk-core/libraries pulled in by torch),
  ROCm PyTorch 2.11 (torch[device-gfx1100]==2.11.0+rocm7.14.0 from
  repo.amd.com/rocm/whl-multi-arch), flash-attn, and the AMD vLLM RDNA wheel via
  uv. Use whenever the user wants to install or set up ROCm + PyTorch + vLLM
  from wheels on a Radeon GPU, or mentions TheRock, virtual-environment ROCm,
  pip-installable ROCm, rocm7.14, whl-multi-arch, repo.amd.com, gfx1100, or asks
  how to build the serving environment on RDNA. Handles the full install flow:
  venv creation, wheel installs, environment variables, and verification. Do not
  use for NVIDIA GPUs, AMD Instinct (use serving-llms-on-instinct), AMD EPYC
  (use serving-llms-on-epyc), or Ryzen AI / NPU.
---

# Install ROCm (TheRock) + PyTorch + vLLM on RDNA

Set up a Python venv with the pip-built ROCm 7.14 stack ("virtual-environment
ROCm" wheels built by TheRock) plus ROCm PyTorch and the AMD vLLM RDNA wheel,
on an AMD RDNA consumer GPU. This is the installation step that produces an
environment ready for serving (see `serving-llms-on-rdna` for the serving
workflow itself).

## Prerequisites

- AMD RDNA GPU: validated on **RX 7900 XTX (gfx1100)**. Other GPUs: use the
  AMD docs configurator (rocm.docs.amd.com, Radeon / vLLM / pip path) to
  generate the matching `device-gfxXXXX` extra and wheel URLs.
- Linux with the amdgpu kernel driver: `/dev/kfd` and `/dev/dri` present.
- A Python 3.14 interpreter (`python3.14`, or a conda env with Python 3.14).
- `uv` installed (`uv --version`). If missing: `pip install uv` or
  `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- ~3 GB free disk: torch + triton + ROCm SDK wheels are ~1.2 GB, the vLLM
  wheel is 395 MB.

## Flow

Follow these steps in order.

### Step 1: Create the venv

The stack is pinned to the Python 3.14 (`cp314`) wheel set.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
```

No `python3.14` on the machine? Create the venv from any Python 3.14
interpreter, e.g. `/path/to/conda/envs/rocm7.14/bin/python -m venv .venv`.
The venv is isolated regardless of which interpreter created it.

**Conda alternative (validated):** install directly into a conda env
instead of a venv — the wheel set is the same. On this repo's target
machine the working env is `rocm7.14`:

```bash
conda create -n rocm7.14 python=3.14 -y
conda activate rocm7.14
# then Steps 2-4 verbatim (pip/uv install into the active env)
```

In conda the PYTHONPATH export (Step 5) uses `$CONDA_PREFIX` instead of
`$VIRTUAL_ENV`:

```bash
export PYTHONPATH=$CONDA_PREFIX/lib/python3.14/site-packages/_rocm_sdk_core/share/amd_smi
```

### Step 2: Install PyTorch (pulls the TheRock ROCm SDK)

```bash
python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "torch[device-gfx1100]==2.11.0+rocm7.14.0" \
    "torchvision[device-gfx1100]==0.26.0+rocm7.14.0" \
    "torchaudio==2.11.0+rocm7.14.0"
```

This also installs the `rocm==7.14.0` meta-package, which pulls the TheRock
virtual ROCm: `rocm-sdk-core`, `rocm-sdk-libraries`, and the device-specific
`rocm-sdk-device-gfx1100` / `amd-torch-device-gfx1100` wheels. No system ROCm
install is needed.

### Step 3: Install flash-attn (RDNA wheel)

```bash
python -m pip install https://rocm.frameworks.amd.com/whl-multi-arch/vllm-rdna/flash-attn/flash_attn-2.8.3-py3-none-any.whl
```

### Step 4: Install vLLM — MUST use `uv pip install`, not pip

```bash
uv pip install https://rocm.frameworks.amd.com/whl-multi-arch/vllm-rdna/vllm/vllm-0.23.1.dev1%2Brocm7.14.0.g9ddef7117.d20260715-cp314-cp314-linux_x86_64.whl
```

**Why uv and not pip:** the vLLM wheel hard-requires `amd-quark>=0.8.99`, and
the only PyPI version satisfying that on Python 3.14 (amd-quark 0.12.post1)
declares `Requires-Python: >=3.11,<3.14` — a wrong metadata annotation; the
package works fine on 3.14. `uv` resolves it anyway; `pip` rejects it and the
install fails with `ERROR: Could not find a version that satisfies the
requirement amd-quark>=0.8.99`. This is exactly the failure the official docs
warn about. Do not substitute pip here.

If the user insists on pip (or uv is unavailable), see
"Installing vLLM with pip instead of uv" in `reference.md` for a
`--no-deps` workaround and its limitation (no Quark quantization).

### Step 5: Set environment variables

```bash
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.14/site-packages/_rocm_sdk_core/share/amd_smi
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
```

`PYTHONPATH` adds the `amdsmi` bindings shipped in the TheRock SDK wheel
(prevents ROCm platform detection errors); `FLASH_ATTENTION_TRITON_AMD_ENABLE`
is required for flash-attn availability on RDNA.

### Step 6: Verify the install

```bash
python scripts/verify.py
```

or manually:

```bash
python -c "import vllm; print('vLLM version:', vllm.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__); print('HIP available:', torch.cuda.is_available())"
python -c "import flash_attn; print('flash-attn:', flash_attn.__version__)"
```

Expected: vLLM `0.23.1.dev1+rocm7.14.0.g9ddef7117.d20260715`, PyTorch
`2.11.0+rocm7.14.0`, `HIP available: True`, flash-attn `2.8.3`.

### Step 7: (Optional) Smoke test

```bash
vllm serve Qwen/Qwen3.5-9B --port 8000
```

Wait for "Application startup complete", then
`curl http://localhost:8000/v1/models` returns the model. See
`serving-llms-on-rdna` for the full serving flow.

Validated on RX 7900 XTX (Aug 2026): `vllm serve Qwen/Qwen2.5-7B-Instruct`
from a local HF cache (no download) — "Application startup complete" in
~40 s, flash-attn active (ROCM_ATTN), steady ~53 tok/s at 8K context,
23.2 GB VRAM with `--gpu-memory-utilization 0.9`. If the model is not
cached locally, `vllm serve` downloads it on first run.

## Version table (as of 2026-08)

| Component | Version | Source |
|-----------|---------|--------|
| Python | 3.14 | venv |
| PyTorch | 2.11.0+rocm7.14.0 | repo.amd.com/rocm/whl-multi-arch |
| torchvision / torchaudio | 0.26.0+rocm7.14.0 / 2.11.0+rocm7.14.0 | same |
| ROCm SDK (TheRock) | rocm 7.14.0 (rocm-sdk-core/libraries) | pulled by torch |
| flash-attn | 2.8.3 | rocm.frameworks.amd.com vllm-rdna |
| vLLM | 0.23.1.dev1+rocm7.14.0 (cp314) | rocm.frameworks.amd.com vllm-rdna |

Newer versions appear as AMD publishes new wheels; regenerate exact commands
from the docs configurator and re-validate with Step 6.
