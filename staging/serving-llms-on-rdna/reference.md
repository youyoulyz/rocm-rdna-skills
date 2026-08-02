# serving-llms-on-rdna — Reference

Deep reference for serving LLMs on AMD RDNA consumer GPUs (RX 6000/7000/9000
series). Covers precision, backends, model selection, known issues, and
troubleshooting. The SKILL.md is the agent playbook; this document is for
non-default decisions.

## RDNA GPU Family Reference

| gfx target | Family | Cards | VRAM | WMMA | HSA override |
|-----------|--------|-------|------|------|--------------|
| gfx1030 | RDNA 2 | RX 6900 XT / 6800 XT | 16 GB | no | 10.3.0 |
| gfx1100 | RDNA 3 | RX 7900 XTX | 24 GB | yes | 11.0.0 |
| gfx1101 | RDNA 3 | RX 7900 XT | 20 GB | yes | 11.0.0 |
| gfx1102 | RDNA 3 | RX 7800 XT / 7900 GRE | 16 GB | yes | 11.0.0 |
| gfx1103 | RDNA 3 | RX 7700 XT / 7600 | 12/8 GB | yes | 11.0.0 |
| gfx1150/1151 | RDNA 3.5 | Ryzen AI APUs | shared | yes | 11.5.0 |
| gfx1200 | RDNA 4 | RX 9070 XT | 16 GB | yes | 12.0.0 |

RDNA 2 (gfx1030) lacks WMMA instructions — llama.cpp must build with
`-DGGML_HIP_ROCWMMA_FATTN=OFF` and vLLM kernel coverage is incomplete.
**Prefer llama.cpp on RDNA 2.** The skill's `gpu_overrides.json` marks
this as `vllm_support: limited`.

## Precision on RDNA

RDNA 3 and 4 support **fp16 and bf16 natively**. FP8 compute exists on
RDNA 4 only (gfx1200); on RDNA 3 it is emulated (dequant to fp16 during
matmul — slower and usually not worth it).

**Not supported at all**: MXFP4/MXFP6 (these are CDNA-4 features) and
NVFP4 (NVIDIA-only). A vLLM checkpoint published in mxfp4 will fail on
RDNA. Convert to bf16 or use GGUF quantized versions instead.

For llama.cpp, GGUF quantization is the practical precision story:
Q4_K_M (~4.5 bits/param) is the default recommendation; IQ4_XS is nearly
identical quality at slightly smaller size. Q5_K_M for quality-critical
work, Q8_0 for near-lossless (only fits for models <= 8B on 24 GB).

## Backend Comparison

| | vLLM (pip) | vLLM (Docker) | llama.cpp (Docker) | llama.cpp (compile) |
|---|---|---|---|---|
| **Install** | `conda run -n torch pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/` | `docker pull rocm/vllm-dev:navi_nightly` | `docker pull ghcr.io/ggml-org/llama.cpp:full-rocm` | cmake build, ~10 min |
| **Weights** | safetensors (HF) | safetensors (HF) | GGUF | GGUF |
| **Quantization** | FP16/BF16 only (FP8 experimental) | same | Q4_K_M etc. | same |
| **Max model (24 GB)** | ~8B FP16 | ~8B FP16 | ~14B Q4_K_M | ~14B Q4_K_M |
| **OpenAI API** | yes | yes | yes (llama-server) | yes |
| **Tool calling** | yes (auto) | yes | yes (newer builds) | yes |
| **Startup** | ~30 s | ~60 s (first pull: minutes) | seconds | seconds |
| **Preferred when** | Python ecosystem integration, best accuracy | reproducible env, no conda | quick GGUF serving, quantization | offline, custom builds |

Both llama.cpp paths need **GGUF files** — vLLM uses safetensors directly.
To get a GGUF model:

```bash
# Option A: huggingface-cli (preferred for official quantized releases)
hf download TheBloke/Qwen2.5-7B-Instruct-GGUF \
  qwen2.5-7b-instruct-q4_k_m.gguf --local-dir ~/models

# Option B: ollama pull (Ollama stores GGUF internally — not directly usable
# by llama.cpp, but Ollama itself is a valid quick path)
ollama pull qwen3:8b
```

Many publishers (Qwen team, Unsloth, bartowski, TheBloke) release official
GGUF quantizations on HF. Prefer official publisher quants over converting
safetensors yourself with `llama-quantize`.

## vLLM on RDNA — Key Environment

Two env vars are mandatory on RDNA consumer GPUs:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # gfx1100; see table above
export VLLM_ROCM_USE_AITER=0             # AITER targets CDNA only; crashes on RDNA
```

`HSA_OVERRIDE_GFX_VERSION` tells the ROCm runtime to treat the consumer
GPU as the compute variant of its architecture (11.0.0 = gfx1100's
compute profile). Without it, ROCm reports "invalid agent" for compute
workloads and every framework fails with `HSA_STATUS_ERROR`.

`VLLM_ROCM_USE_AITER=0` disables the AITER optimization package that
ships with vLLM's ROCm wheels. AITER kernels are tuned for MI300/MI350
CDNA hardware; enabling them on RDNA crashes at kernel load.

Also recommended: `--enforce-eager` (skip HIP graph capture, which can
OOM on 24 GB when the model uses most of VRAM — eager mode is ~10%
slower but far more reliable).

### pip install

```bash
conda activate torch
pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
```

Version-pinned install:
```bash
pip install vllm==0.18.0+rocm721 --extra-index-url https://wheels.vllm.ai/rocm/0.18.0/rocm721
```

If the installed vLLM is built for an older torch, `pip install` will try
to downgrade torch — that is fine as long as the ROCm torch index is
used. Do **not** mix with the CUDA torch wheel.

CLI note: `vllm serve` may not be on PATH in a pip install. Use the
module form:

```bash
python -m vllm.entrypoints.openai.api_server --model <id> --port 8000 ...
```

### Docker

Use the RDNA ("navi") images — the generic `vllm/vllm-openai-rocm` image
is built for CDNA and may lack gfx1100 kernels. `rocm/vllm-dev` with a
`navi` tag is the community-standard choice for 7900 XTX:

```bash
docker pull rocm/vllm-dev:navi_nightly

docker run -d --name vllm-rdna \
  --device /dev/kfd --device /dev/dri \
  --group-add=video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  -e HF_TOKEN=${HF_TOKEN} \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  -e VLLM_ROCM_USE_AITER=0 \
  rocm/vllm-dev:navi_nightly \
  --model Qwen/Qwen3-8B --port 8000 \
  --gpu-memory-utilization 0.9 --enforce-eager
```

## llama.cpp on RDNA

### Compile from source

```bash
sudo apt install git wget hipcc libhipblas-dev librocblas-dev cmake build-essential

git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DAMDGPU_TARGETS=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

- `AMDGPU_TARGETS` should list your exact gfx target (gfx1100 for
  7900 XTX). Building all targets multiplies compile time.
- `GGML_HIP_ROCWMMA_FATTN=ON` enables flash attention via rocWMMA
  (RDNA 3+ only; **OFF on RDNA 2**).
- Missing `libhipblas-dev` makes CMake silently skip the HIP backend —
  always check the configure output for "HIP" in the enabled backends.
- Add `-DLLAMA_CURL=ON` to download models directly with llama.cpp.

### Serve with llama-server

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 ./build/bin/llama-server \
  -m ~/models/<model>.gguf \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --host 0.0.0.0 --port 8080
```

`--n-gpu-layers 99` offloads all layers to the GPU. For partial offload
(large models), use `--n-gpu-layers 24` — the remainder runs on CPU.
On the 7900 XTX a 14B Q4_K_M model with full offload runs at roughly
30-45 tokens/s; an 8B at 50-70 tokens/s.

### Docker

```bash
docker pull ghcr.io/ggml-org/llama.cpp:full-rocm

docker run -d --name llama-rdna \
  --device /dev/kfd --device /dev/dri \
  --group-add=video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --ipc=host \
  -v ~/models:/models -p 8080:8080 \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  ghcr.io/ggml-org/llama.cpp:full-rocm \
  -m /models/<model>.gguf --ctx-size 32768 \
  --n-gpu-layers 99 --host 0.0.0.0 --port 8080
```

Alternative AMD-maintained image: `rocm/llama.cpp:latest` (tag format
`llama.cpp-<version>_rocm<ver>_ubuntu<ver>_full`).

## Model Selection for 24 GB (7900 XTX)

Tier list by what fits and is worth running:

| Tier | Models | Why |
|------|--------|-----|
| Best quality | Qwen3-14B (Q4_K_M), Gemma-3-12b (Q4_K_M) | ~8-9 GB weights, huge KV headroom |
| Best all-round | Qwen3-8B (FP16 vLLM or Q4 llama.cpp) | FP16 works in vLLM with ~27K ctx; Q4 leaves room for 128K |
| Reasoning | DeepSeek-R1-Distill-Qwen-8B, Qwen3-8B-Thinking | Same footprint as Qwen3-8B |
| Long context | Qwen3-4B (FP16, full 128K) | Small enough for max context at full precision |
| Vision | Qwen2.5-VL-7B-Instruct (vLLM) | VLM support is weaker in llama.cpp |
| Compact/fast | Phi-4-mini-instruct, Llama-3.2-3B | Instant startup, high throughput |

Will not fit even at Q4: 70B+ dense models, DeepSeek-V3 (671B MoE).
On 24 GB, 14B Q4_K_M is the practical ceiling for comfortable context.

Quantization sizing (24 GB, llama.cpp): 14B Q4_K_M ≈ 8 GB weights →
~12 GB KV headroom ≈ full 128K context. 32B Q4_K_M ≈ 20 GB weights →
~1.5 GB KV headroom ≈ 2-4K context (usable, but cramped).

## Known Issues and Workarounds

| Symptom | Cause | Fix |
|---------|-------|-----|
| `HSA_STATUS_ERROR_INVALID_AGENT` / "No agents found" | Missing HSA_OVERRIDE_GFX_VERSION | `export HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| vLLM crashes at startup on RDNA | AITER kernels targeting CDNA | `export VLLM_ROCM_USE_AITER=0` |
| OOM during vLLM startup on a model that should fit | HIP graph capture peak | `--enforce-eager` |
| vLLM pip: `undefined symbol: _ZN3c103hip19getCurrentHIPStreamEa` warnings followed by `AttributeError: '_C' object has no attribute 'rms_norm'` and `EngineCore failed to start` | **Fatal ABI mismatch** — vLLM wheel built for a different Python (cp312 wheel force-installed into cp310) and/or different ROCm series (rocm700 wheel with torch rocm7.2). The `_C` extension never loads, so core kernels are missing. | Install a wheel matching your Python and ROCm. Best path: fresh conda env with **Python 3.12** (`conda create -n vllm python=3.12`) then `pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/` (ROCm wheels on wheels.vllm.ai are cp312-only). Or use the Docker backend. |
| vLLM Docker: MoE model (gpt-oss-20b) crashes with `TritonAMDGPUToLLVM/MFMA.cpp:869: Assertion ... DotOperand layout` during startup | vLLM's Triton MoE kernels (conch-triton-kernels) target MFMA, which is CDNA-only. RDNA has no MFMA instructions. Dense models are unaffected. | Serve MoE models with **llama.cpp** (GGUF) instead. Dense models (Llama-3-8B etc.) work fine in vLLM. |
| vLLM logs `Repo id must be in the form 'repo_name' or 'namespace/repo_name'` when `--model` is a local path | vLLM 0.11.x dev bug: safetensors-metadata lookup passes local paths to the HF API | **Non-fatal** — caught and ignored. Proceeds with local loading. |
| llama.cpp Docker: `cudaMalloc failed: out of memory` at model load | Another service (e.g. vLLM) already holds most of the 24 GB VRAM. 24 GB is too small for two models. | Run one serving process at a time. Check `rocm-smi --showmeminfo vram`. |
| llama.cpp builds but GPU not used (all layers on CPU) | HIP backend silently skipped | Rebuild with `-DGGML_HIP=ON` and confirm "HIP" in cmake output; install `libhipblas-dev` |
| Slow generation vs. NVIDIA reports | RDNA 3 lacks fp8 fast path; WMMA is narrower | Use Q4_K_M; accept ~40-70 tok/s range for 7-14B |
| `HSA_OVERRIDE_GFX_VERSION` wrong value | e.g. 11.0.0 on RDNA 2 | Use 10.3.0 for gfx1030 |
| Docker can't see GPU | Missing --device or group perms | `--device /dev/kfd --device /dev/dri --group-add=video`; user in video/render group |
| `CUDA_VISIBLE_DEVICES=''` | Hides all GPUs from ROCm runtime | `unset CUDA_VISIBLE_DEVICES` |

### Verified on RX 7900 XTX (ROCm 7.2.4, July 2026)

- vLLM Docker + dense model (Meta-Llama-3-8B-Instruct, local safetensors):
  loads, serves, responds. Health OK.
- llama.cpp Docker (`rocm/llama.cpp:..._server`) + Qwen2.5-7B-Instruct-Q4_K_M
  (local GGUF): GPU detected (`Device 0: AMD Radeon RX 7900 XTX, gfx1100`),
  loads, serves, responds.
- vLLM pip with a mismatched wheel (cp312 rocm700 wheel in cp310 env with
  torch rocm7.2) fails fatally — see ABI row above.
- vLLM Docker + MoE model (gpt-oss-20b) fails at Triton kernel compile —
  see MFMA row above. Use llama.cpp for MoE.

## Alternative Quick Path: Ollama

Ollama (if installed) is the zero-config path: it bundles its own ROCm
support, handles HSA_OVERRIDE internally, and manages GGUF downloads.

```bash
ollama run qwen3:8b          # interactive
ollama serve                 # OpenAI-compatible endpoint on :11434
```

This skill's purpose is vLLM and llama.cpp with explicit control. If the
user just wants "run a model quickly", Ollama is the right answer and
this skill should say so rather than spinning up a full serving stack.

## What This Skill Does NOT Cover

- **Instinct/CDNA GPUs** (MI300X etc.) — use `serving-llms-on-instinct`
- **EPYC servers** — use `serving-llms-on-epyc`
- **Ryzen AI APUs / NPU** — use `local-ai-use` / `apu-memory-tuner`
- **NVIDIA GPUs** — different drivers, different everything
- **Training / fine-tuning** — this skill is inference-only
- **Multi-GPU** — RDNA consumer boards are single-GPU; no tensor parallelism
- **Remote hosts** — consumer desktops are local-only
