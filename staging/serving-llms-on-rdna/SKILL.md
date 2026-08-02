---
name: serving-llms-on-rdna
description: >-
  Serves AI models on AMD RDNA consumer GPUs (RX 7900 XTX, RX 7900 XT, RX 6900 XT,
  RX 9070 XT, etc.) using vLLM or llama.cpp. Use this skill whenever the user wants
  to run, serve, deploy, start, host, or launch a language model on an AMD consumer
  GPU, AMD Radeon, RX 7000 series, RX 6000 series, gfx1100, or gfx1030. Also use
  when the user mentions vLLM on RDNA, llama.cpp on AMD, serving on a consumer AMD
  GPU, Ollama model serving, or asks how to get a model running on their AMD Radeon
  card. Handles the full flow: GPU detection, environment validation, model
  selection under VRAM constraints, launch (vLLM pip / vLLM Docker / llama.cpp
  Docker / llama.cpp compile), and health verification. Do not use for NVIDIA GPUs,
  AMD Instinct (MI300X/MI325X/MI350X — use serving-llms-on-instinct), AMD EPYC
  (use serving-llms-on-epyc), Ryzen AI APUs, or NPU.
---

# Serving LLMs on RDNA Consumer GPUs

Serve an LLM as an OpenAI-compatible endpoint on an AMD RDNA consumer GPU
(RX 6000/7000/9000 series). Supports four backends:

| Backend | Install method | When to prefer |
|---------|---------------|----------------|
| **vllm-pip** | `pip install vllm` in conda env | Python ecosystem, best accuracy, tool calling |
| **vllm-docker** | `rocm/vllm-dev:navi_*` image | Reproducible env, no conda |
| **llama-cpp-docker** | `ghcr.io/ggml-org/llama.cpp:full-rocm` | Quick GGUF serving, quantization |
| **llama-cpp-compile** | cmake build from source | Offline, custom builds |

This is a **single-GPU, local-only** skill: consumer desktops, no tensor
parallelism, no SSH remoting. Endpoint is `localhost:<port>`.

## Prerequisites

- AMD RDNA GPU (RX 6000/7000/9000 series) with ROCm installed
- Linux (Ubuntu/Debian tested)
- One of: conda env with ROCm PyTorch (vllm-pip), Docker (vllm-docker /
  llama-cpp-docker), or cmake + hipcc (llama-cpp-compile)

## Flow

Follow these steps in order. Do not skip ahead; each step's output feeds
the next.

### Step 1: Detect the GPU

Run `scripts/detect.py`. It reports the gfx version, RDNA family, VRAM,
and the required `HSA_OVERRIDE_GFX_VERSION` value.

```bash
uv run --quiet scripts/detect.py --json
```

- Exit 0: RDNA GPU found — proceed.
- Exit 1: no GPU / amd-smi failed — report the hint, stop.
- Exit 2: AMD GPU but not RDNA (e.g. gfx942) — route to
  `serving-llms-on-instinct`, stop.
- Note the `expected_hsa_override` value — you'll need it in Step 7.
- If `gpu_family` is `rdna2` (gfx1030), strongly prefer llama.cpp:
  vLLM kernel coverage is incomplete on RDNA 2.

### Step 2: Validate the Environment

Run `scripts/validate.py` to check prerequisites for all backends:

```bash
uv run --quiet scripts/validate.py --json
```

The output has `backends_available` — a map of which of the four
backends are ready. If your target backend reports `false`, check the
`errors` array for what blocks it and the `fix` for each.

Rules:
- If all four backends are `false`, report the blocking errors and stop.
- A `warning` degrades performance but does not block.
- `HSA_OVERRIDE_GFX_VERSION not set` warning: expected; the launch
  command sets it explicitly (Step 7).

### Step 3: Choose the Backend

Present the available backends to the user in this order, recommending
the first that is available:

1. **vllm-pip** — already installed in the conda `torch` env on most
   setups; no new install needed
2. **llama-cpp-compile** — only ~10 min build, no huge image downloads
3. **vllm-docker** — reproducible but ~16 GB image
4. **llama-cpp-docker** — lightweight (~2-3 GB image)

If the user only wants "run a model quickly" without an endpoint, suggest
Ollama (see reference.md) instead — this skill is for explicit vLLM /
llama.cpp serving.

### Step 4: Select the Model

**First check for local models** — if the user has a local model
directory (e.g. `/mnt/public/models` for safetensors, `/mnt/public/GGUF_models`
for GGUF), prefer local files over downloading. No network needed, no
HF gating. vLLM takes a directory of safetensors; llama.cpp takes a
`.gguf` file path.

If the user names a model, use it. Otherwise consult `data/model_guide.json`:

- Use the `default_model` (`Qwen/Qwen3-8B`) unless the user wants a
  specific capability (reasoning, vision, long context).
- Check `will_not_fit` — if the user's model is listed, refuse politely
  and suggest the closest alternative from the `models` list.
- For models in the guide, the estimated sizes are already there. For
  any other model, go to Step 5.
- **GGUF note**: llama.cpp backends need GGUF weights. Check the guide /
  ask the user for the GGUF path, or download it (see reference.md —
  prefer official publisher quants like `Qwen/Qwen3-14B-GGUF`).
- **MoE note**: MoE models (gpt-oss-20b, Qwen3.6-35B-A3B, LFM2.5,
  OLMoE) crash in vLLM on RDNA — the Triton MoE kernels target CDNA's
  MFMA instructions. Serve MoE models with **llama.cpp** (GGUF). vLLM is
  reliable for dense models only.
- **One model at a time**: 24 GB holds one serving process. A second
  backend will OOM at model load (llama.cpp reports `cudaMalloc failed`).

### Step 5: Estimate VRAM

Run `scripts/estimate_vram.py` with the detected VRAM (24 GB for 7900 XTX):

```bash
# vLLM backend (safetensors weights)
uv run --quiet scripts/estimate_vram.py --model-id <id> --vram-gb 24 --backend vllm

# llama.cpp backend with quantization
uv run --quiet scripts/estimate_vram.py --model-id <id> --vram-gb 24 \
  --backend llama-cpp --quantization Q4_K_M
```

Interpret the `fit.verdict`:

- **fit** — proceed as planned.
- **tight** — proceed but limit `--max-model-len` to
  `fit.recommended_max_model_len`; mention quantization as an option.
- **refuse** — the model cannot run at the requested precision. If
  backend is vllm, suggest llama.cpp with Q4_K_M; if already llama.cpp,
  suggest a smaller model from `model_guide.json`. Do not attempt to
  launch a refused model.

If the user's model is not in `model_guide.json` and estimate fails
(offline / gated), proceed with the user's request but note the estimate
is unverified and keep context conservative (8K).

### Step 6: Confirm the Plan

Present a summary table and wait for explicit user approval:

| Field | Value |
|-------|-------|
| Model | `Qwen/Qwen3-8B` |
| Backend | vllm-pip |
| Precision | fp16 (or Q4_K_M) |
| Weight VRAM | 15.3 GB |
| Context limit | 27648 |
| Port | 8000 |

Include any warnings (tight fit, first-run download size). Do not
launch before approval.

### Step 7: Launch

Get the GPU config from `data/gpu_overrides.json` (gfx version → env
defaults, HSA override, docker flags). The two env vars below are
**mandatory** on RDNA. Substitute `<ctx>` with the confirmed context
length and `<port>` with the confirmed port (default 8000).

#### vllm-pip (conda env)

```bash
conda activate torch
HSA_OVERRIDE_GFX_VERSION=<override> VLLM_ROCM_USE_AITER=0 \
python -m vllm.entrypoints.openai.api_server \
  --model <model-id> \
  --port <port> \
  --gpu-memory-utilization 0.9 \
  --max-model-len <ctx> \
  --enforce-eager
```

(Use `python -m vllm.entrypoints.openai.api_server`, not `vllm serve` —
the CLI entry point is often missing in pip installs.)

> **pip install prerequisite**: ROCm vLLM wheels on wheels.vllm.ai are
> **Python 3.12 only** (`cp312`). If your conda env is not 3.12, the
> wheel will fail with ABI errors at startup (`'_C' object has no
> attribute 'rms_norm'`). Create a fresh env:
> `conda create -n vllm python=3.12 && conda activate vllm && pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/`
> and make sure the installed wheel's ROCm series (rocm721 etc.) matches
> your torch (check `pip show torch` → `+rocm7.2`).

#### vllm-docker

```bash
docker run -d --name vllm-rdna \
  --device /dev/kfd --device /dev/dri \
  --group-add=video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p <port>:8000 \
  -e HF_TOKEN=${HF_TOKEN} \
  -e HSA_OVERRIDE_GFX_VERSION=<override> \
  -e VLLM_ROCM_USE_AITER=0 \
  rocm/vllm-dev:navi_nightly \
  --model <model-id> --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len <ctx> \
  --enforce-eager
```

#### llama-cpp-docker

Requires a GGUF file. Mount the directory containing it:

```bash
docker run -d --name llama-rdna \
  --device /dev/kfd --device /dev/dri \
  --group-add=video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --ipc=host \
  -v ~/models:/models \
  -p <port>:8080 \
  -e HSA_OVERRIDE_GFX_VERSION=<override> \
  ghcr.io/ggml-org/llama.cpp:full-rocm \
  -m /models/<model>.gguf \
  --ctx-size <ctx> \
  --n-gpu-layers 99 \
  --host 0.0.0.0 --port 8080
```

#### llama-cpp-compile

Build once (only if `~/llama.cpp/build/bin/llama-server` does not exist):

```bash
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DAMDGPU_TARGETS=<gfx_target> \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

(For RDNA 2 / gfx1030, use `-DGGML_HIP_ROCWMMA_FATTN=OFF`.)

Then serve:

```bash
HSA_OVERRIDE_GFX_VERSION=<override> ~/llama.cpp/build/bin/llama-server \
  -m ~/models/<model>.gguf \
  --ctx-size <ctx> \
  --n-gpu-layers 99 \
  --host 0.0.0.0 --port <port>
```

### Step 8: Verify

Poll the health endpoint until ready:

```bash
# vLLM (returns 503 while loading, 200 when ready)
curl -sf http://localhost:<port>/health

# llama.cpp (returns 200 when ready)
curl -sf http://localhost:<port>/health
```

- Check for port conflicts first: `ss -tlnp | grep <port>`.
- Docker: check `docker logs <name>` if the container exited.
- Poll up to 5 minutes for vLLM (model download + kernel compilation),
  up to 2 minutes for llama.cpp.

Once healthy, send a warmup request to trigger any deferred kernel
compilation:

```bash
curl -s http://localhost:<port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"say hi"}],"max_tokens":5}'
```

Then present the connection table:

| Field | Value |
|-------|-------|
| Model | Qwen/Qwen3-8B |
| Base URL | `http://localhost:<port>/v1` |
| Port | 8000 |
| Backend | vllm-pip |
| Context | 27648 |
| GPU | Radeon RX 7900 XTX (gfx1100, 24 GB) |

And a ready-to-run example:

```bash
curl -s http://localhost:<port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"Hello!"}]}'
```

## Troubleshooting

If launch or health check fails, consult `reference.md` — Known Issues
table. Common cases on RDNA:

- `HSA_STATUS_ERROR_INVALID_AGENT` → `HSA_OVERRIDE_GFX_VERSION` missing
  or wrong.
- vLLM crash at startup → `VLLM_ROCM_USE_AITER` not set to 0.
- OOM at startup → add `--enforce-eager` or reduce context.
- llama.cpp GPU not used → rebuild with HIP backend confirmed in cmake
  output.

If the endpoint still fails after checking these, report the error
message and stop. Do not retry endlessly.

## Verification Checklist

- [ ] `detect.py` exits 0 and reports the RDNA gfx version
- [ ] `validate.py` reports the chosen backend available
- [ ] `estimate_vram.py` verdict is `fit` or `tight` (not `refuse`)
- [ ] User approved the plan table
- [ ] `/health` returns 200
- [ ] Warmup chat completion returns a response
- [ ] Connection table delivered with curl example
