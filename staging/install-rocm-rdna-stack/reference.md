# Reference: ROCm (TheRock) + PyTorch + vLLM on RDNA — install details

Deep reference for the install flow. The `SKILL.md` is the primary entry;
this file holds the why and the edge cases.

## What "TheRock virtual ROCm" means here

ROCm 7.14 is consumed entirely as pip wheels instead of a system
`/opt/rocm` install:

- `rocm==7.14.0` — meta-package (`rocm[libraries]` extra pulls the rest).
- `rocm-sdk-core==7.14.0` — ROCm runtime + `amdsmi` Python bindings
  (at `site-packages/_rocm_sdk_core/share/amd_smi`).
- `rocm-sdk-libraries==7.14.0` — rocBLAS, MIOpen, RCCL, hipBLASLt, etc.
- `rocm-sdk-device-gfx1100==7.14.0` + `amd-torch-device-gfx1100` /
  `amd-torch-device-gfx110x` — device-specific torch prebuilt kernels.
- `rocm-bootstrap` — runtime env helpers.

These wheels are built by TheRock (github.com/rocm/therock, "The HIP
Environment and ROCm Kit"). Consequences:

- No system ROCm, no `apt` ROCm packages, no `rocm-smi` binary needed.
- The venv must stay active for runtime library resolution.
- `PYTHONPATH` must include `.../share/amd_smi` (Step 5 of SKILL.md) or
  `import vllm` / platform detection can fail with amd-smi errors.

## Why the vLLM install must use uv (the amd-quark trap)

Facts verified 2026-08 on Python 3.14.0:

- The vLLM cp314 wheel (vllm-0.23.1.dev1+rocm7.14.0, 395 MB) hard-requires
  `amd-quark>=0.8.99` in its METADATA.
- PyPI amd-quark versions: 0.9 / 0.10 / 0.11.x declare `Requires-Python:
  <3.13`; **0.12.post1 declares `>=3.11,<3.14` but actually imports and
  runs fine on 3.14** (verified: `import quark` → 0.12.post1).
- pip honors the declared `Requires-Python` and rejects 0.12.post1 on
  3.14, so resolution fails with `No matching distribution found for
  amd-quark>=0.8.99` — from any index, including pypi.org (the aliyun
  mirror additionally only carries amd-quark ≤ 0.6.0).
- uv accepts 0.12.post1 (it does not treat the bad annotation as fatal),
  installs the full dependency graph, and the result works.

Net effect: `uv pip install <vllm-wheel-url>` succeeds; `pip install
<vllm-wheel-url>` fails. This is what the official docs' warning ("pip may
silently pull incompatible versions from PyPI when installing from a direct
wheel URL") refers to.

If/when AMD publishes a corrected amd-quark (0.13+ supporting 3.14), pip
will work and this trap disappears — re-test before removing the uv
requirement from the skill.

## Installing vLLM with pip instead of uv (workaround)

Only if uv is truly unavailable. The wheel's other 75 base dependencies
install fine from PyPI; only the amd-quark pin is unsatisfiable.

```bash
# 1. Install the wheel without dependency resolution
pip install --no-deps /path/to/vllm-0.23.1.dev1+rocm7.14.0...cp314-linux_x86_64.whl

# 2. Install every base dependency EXCEPT amd-quark
#    (extract Requires-Dist from the wheel METADATA, drop amd-quark,
#     drop lines with "; extra ==", install the rest from pypi.org)
pip install --index-url https://pypi.org/simple -r vllm_deps.txt
```

Verified: `import vllm` works with this setup — vllm never imports
`amd-quark` at module load; `quark.torch.*` is only imported lazily inside
MXFP4/MXFP6 quantization helper functions, each guarded by
`try/except ImportError` with a clear "please install amd-quark" message.

Limitation: Quark-quantized models (e.g. MXFP4 checkpoints) will not work —
they hit the lazy import and raise. `uv`'s path installs amd-quark 0.12.post1
and has no such limitation. Prefer uv.

## Environment variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYTHONPATH` | `$VIRTUAL_ENV/lib/python3.14/site-packages/_rocm_sdk_core/share/amd_smi` | amdsmi Python bindings from TheRock SDK (ROCm platform detection) |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `TRUE` | flash-attn availability on RDNA |

Both must be exported before starting Python/vLLM. In conda environments
substitute the conda env site-packages path for `$VIRTUAL_ENV`.

## GPU variants (beyond the validated gfx1100 path)

The validated path uses the `device-gfx1100` extra (RX 7900 XTX, Navi 31;
also covers RX 7900 XT / GRE). The `whl-multi-arch` repo hosts other device
variants (e.g. `amd-torch-device-gfx110x`) and the vllm-rdna directory may
gain wheels for other gfx versions over time. For an unvalidated GPU:

1. Open the AMD docs configurator: rocm.docs.amd.com → AI ecosystem →
   vLLM → Radeon / pip path, and select the GPU.
2. Use the generated `device-gfxXXXX` extras and wheel URLs.
3. Run Step 6 verification before relying on it.

gfx1100 needs no `HSA_OVERRIDE_GFX_VERSION`. Older RDNA (gfx1030) needs an
override for torch/vllm and is NOT covered by this install path — use
`serving-llms-on-rdna`'s llama.cpp backends instead.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No matching distribution found for amd-quark>=0.8.99` | pip used instead of uv (see above) | Use `uv pip install`; or the `--no-deps` workaround |
| `import vllm` fails on amd-smi / platform detection | `PYTHONPATH` missing | Export the amd_smi path (Step 5) |
| flash-attn unavailable at runtime | `FLASH_ATTENTION_TRITON_AMD_ENABLE` missing | Export `TRUE` |
| `ImportError: amd-quark required ... MX-FP4` at model load | amd-quark absent (pip workaround path) | Reinstall via uv, or avoid Quark-quantized models |
| pip resolves torch from pypi.org (no rocm build) | `--index-url` dropped | Keep `--index-url https://repo.amd.com/rocm/whl-multi-arch/` |
| `python3.14` not found | no system 3.14 | Create venv from any Python 3.14 (conda env works) |
