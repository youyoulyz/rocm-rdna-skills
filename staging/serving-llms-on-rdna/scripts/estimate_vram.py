#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Estimate VRAM requirements for a HuggingFace model on RDNA GPU.

Reports weight memory (from safetensors metadata) and KV cache per token
(from model config). With --vram-gb, estimates achievable context length
and whether the model fits.

Supports both vLLM (safetensors weights) and llama.cpp (GGUF quantized)
backends. For llama.cpp, pass --quantization to estimate quantized weight
sizes instead of FP16.

Usage:
    python scripts/estimate_vram.py --model-id Qwen/Qwen3-8B --vram-gb 24
    python scripts/estimate_vram.py --model-id Qwen/Qwen3-14B --vram-gb 24 --backend llama-cpp --quantization Q4_K_M

Output: JSON to stdout. Exits 0 on success, 1 on failure.

Env vars:
    HF_TOKEN -- required for gated/private models
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
import urllib.error

HF_BASE = "https://huggingface.co"

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
    "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}

# llama.cpp GGUF quantization: bits per parameter (includes scale factors)
QUANT_BITS = {
    "Q4_K_M": 4.5,
    "Q4_K_S": 4.25,
    "IQ4_XS": 4.25,
    "Q5_K_M": 5.5,
    "Q5_K_S": 5.25,
    "Q8_0": 8.5,
    "FP16": 16,
    "FP8": 8,
}


def _fetch(url, token=None):
    headers = {"User-Agent": "estimate-vram-rdna/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return None, "Authentication required. Set HF_TOKEN for gated/private models."
        if e.code == 403:
            return None, "Access denied. Accept the model license at huggingface.co."
        if e.code == 404:
            return None, "Not found."
        return None, f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        return None, str(e)


def _weight_memory(model_id, revision, token):
    """Get weight memory in bytes.

    Uses two signals and picks the more reliable one:
      1. safetensors metadata (dtype x param count) from the model info API
      2. raw .safetensors file sizes from the tree API

    For standard BF16/FP16 checkpoints both agree. For quantized models
    (QAT, GPTQ, AWQ) the metadata reports packed INT32 containers while the
    actual data is 4-bit, making (1) vastly overestimate. File sizes are
    always ground truth because safetensors is uncompressed, so when both
    are available we take the smaller value.

    Also returns param_count (FP16-equivalent) for GGUF quantization
    estimation.
    """
    metadata_bytes = 0
    file_bytes = 0
    param_count = 0

    # Signal 1: safetensors dtype x count from model info API
    url = f"{HF_BASE}/api/models/{model_id}?expand[]=safetensors"
    info, err = _fetch(url, token)
    if info:
        params = info.get("safetensors", {}).get("parameters", {})
        if params:
            metadata_bytes = sum(
                count * DTYPE_BYTES.get(dtype, 2)
                for dtype, count in params.items()
            )
            param_count = sum(count for dtype, count in params.items())

    # Signal 2: raw file sizes from tree API
    tree_url = f"{HF_BASE}/api/models/{model_id}/tree/{revision}"
    entries, tree_err = _fetch(tree_url, token)
    if entries and isinstance(entries, list):
        file_bytes = sum(
            e.get("size", 0) for e in entries
            if e.get("type") == "file"
            and e.get("path", "").endswith(".safetensors")
        )

    # Pick the best estimate
    if metadata_bytes and file_bytes:
        if file_bytes < metadata_bytes * 0.8:
            # Large gap means quantized weights packed in wider containers.
            return file_bytes, "file_sizes", param_count, None
        return metadata_bytes, "safetensors_metadata", param_count, None
    if metadata_bytes:
        return metadata_bytes, "safetensors_metadata", param_count, None
    if file_bytes:
        return file_bytes, "file_sizes", param_count, None

    return 0, None, 0, err or tree_err or "No safetensors files found"


def _model_config(model_id, revision, token):
    """Fetch config.json. Handles nested configs (VLMs, multimodal)."""
    url = f"{HF_BASE}/{model_id}/resolve/{revision}/config.json"
    config, err = _fetch(url, token)
    if not config:
        return None, err

    # Some multimodal models nest the LLM config under a sub-key
    if "num_hidden_layers" not in config:
        for key in ("text_config", "language_config", "llm_config"):
            sub = config.get(key, {})
            if "num_hidden_layers" in sub:
                # Merge sub-config but keep top-level max_position_embeddings
                max_seq = config.get("max_position_embeddings",
                                     sub.get("max_position_embeddings"))
                merged = dict(sub)
                if max_seq:
                    merged["max_position_embeddings"] = max_seq
                return merged, None

    return config, None


def _kv_per_token(config):
    """KV cache bytes per token at BF16. Returns (bytes, details)."""
    if not config:
        return 0, {}

    n_layers = config.get("num_hidden_layers", 0)
    if not n_layers:
        return 0, {}

    n_kv = config.get("num_key_value_heads",
                      config.get("num_attention_heads", 0))
    hdim = config.get("head_dim", 0)
    if not hdim:
        hsz = config.get("hidden_size", 0)
        n_heads = config.get("num_attention_heads", 1)
        hdim = hsz // n_heads if n_heads else 0

    details = {"num_layers": n_layers, "num_kv_heads": n_kv, "head_dim": hdim}

    # MLA (DeepSeek-R1/V3): compressed KV via latent projection
    if "kv_lora_rank" in config:
        kv_rank = config["kv_lora_rank"]
        rope_dim = config.get("qk_rope_head_dim", 0)
        kv = 2 * n_layers * (kv_rank + rope_dim) * 2
        details.update(mla=True, kv_lora_rank=kv_rank, qk_rope_head_dim=rope_dim)
        return kv, details

    # Standard: 2 (K+V) * layers * kv_heads * head_dim * 2 bytes (bf16)
    return 2 * n_layers * n_kv * hdim * 2, details


def _quantized_weight_gb(fp16_gb, quant):
    """Estimate GGUF quantized weight size from FP16 size and quant bits."""
    bits = QUANT_BITS.get(quant)
    if not bits:
        return None
    return round(fp16_gb * bits / 16, 1)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-id", required=True, help="HuggingFace model ID")
    p.add_argument("--revision", default="main")
    p.add_argument("--vram-gb", type=float, default=0,
                   help="GPU VRAM in GB (enables fit check)")
    p.add_argument("--backend", choices=["vllm", "llama-cpp"], default="vllm")
    p.add_argument("--quantization", default="",
                   help="GGUF quantization for llama.cpp backend "
                        "(Q4_K_M, IQ4_XS, Q5_K_M, Q8_0, FP16)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN", "")

    # Weight memory
    w_bytes, source, param_count, err = _weight_memory(
        args.model_id, args.revision, token
    )
    if not w_bytes:
        print(json.dumps({"error": f"Cannot estimate weight memory: {err}",
                          "model_id": args.model_id}))
        sys.exit(1)

    fp16_gb = round(w_bytes / (1024**3), 1)

    # Model config and KV cache
    config, _ = _model_config(args.model_id, args.revision, token)
    kv_bytes, kv_details = _kv_per_token(config)
    max_seq = config.get("max_position_embeddings") if config else None

    result = {
        "model_id": args.model_id,
        "backend": args.backend,
    }

    if args.backend == "llama-cpp" and args.quantization:
        w_gb = _quantized_weight_gb(fp16_gb, args.quantization)
        if w_gb is None:
            print(json.dumps({"error": f"Unknown quantization {args.quantization}",
                              "known": sorted(QUANT_BITS)}))
            sys.exit(1)
        result["weight_memory_gb"] = w_gb
        result["weight_fp16_gb"] = fp16_gb
        result["quantization"] = args.quantization
        result["source"] = "quantization_estimate"
    else:
        result["weight_memory_gb"] = fp16_gb
        result["source"] = source

    if kv_bytes:
        result["kv_cache_bytes_per_token"] = kv_bytes
        result["kv_cache"] = kv_details
    if max_seq:
        result["model_max_seq_len"] = max_seq

    # Fit estimation
    if args.vram_gb > 0:
        util = args.gpu_memory_utilization
        w_gb = result["weight_memory_gb"]
        usable = round(args.vram_gb * util, 1)
        # Overhead: vLLM activations + internal buffers (~2.5 GB on RDNA);
        # llama.cpp context buffers + scratch (~1.5 GB).
        overhead = 2.5 if args.backend == "vllm" else 1.5
        remaining = round(max(0, usable - w_gb - overhead), 1)

        fit = {
            "gpu_vram_gb": args.vram_gb,
            "gpu_memory_utilization": util,
            "weight_gb": w_gb,
            "usable_vram_gb": usable,
            "overhead_gb": overhead,
            "remaining_for_kv_gb": remaining,
            "weights_fit": w_gb < usable,
        }

        if kv_bytes > 0 and remaining > 0:
            rem_bytes = remaining * (1024**3)
            ctx_bf16 = int(rem_bytes / kv_bytes)
            ctx_fp8 = int(rem_bytes / (kv_bytes / 2))

            fit["max_seq_len_bf16_kv"] = ctx_bf16
            fit["max_seq_len_fp8_kv"] = ctx_fp8

            if max_seq:
                rec = min(ctx_bf16, max_seq)
                rec = (rec // 1024) * 1024
                fit["recommended_max_model_len"] = rec
                fit["context_limited"] = ctx_bf16 < max_seq

        # Fit verdict
        if not fit["weights_fit"]:
            fit["verdict"] = "refuse"
            fit["verdict_reason"] = (
                f"Weights {w_gb} GB exceed usable VRAM {usable} GB. "
                "Use a smaller model or quantize (llama.cpp backend)."
            )
        elif remaining < 1.5:
            fit["verdict"] = "refuse"
            fit["verdict_reason"] = (
                f"Only {remaining} GB left for KV cache after overhead — "
                "will OOM at any useful context. Use a smaller model or quantize."
            )
        elif remaining < 4:
            fit["verdict"] = "tight"
            fit["verdict_reason"] = (
                f"{remaining} GB left for KV cache. Limit context to "
                f"{fit.get('recommended_max_model_len', 2048)} tokens. "
                "Consider quantization for more headroom."
            )
        else:
            fit["verdict"] = "fit"
            fit["verdict_reason"] = (
                f"{remaining} GB left for KV cache — comfortable."
            )

        result["fit"] = fit

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
