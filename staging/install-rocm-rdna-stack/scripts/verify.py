#!/usr/bin/env python3
"""Verify the ROCm (TheRock) + PyTorch + vLLM install on RDNA.

Run from inside the activated venv, after exporting:
  PYTHONPATH=$VIRTUAL_ENV/lib/python3.14/site-packages/_rocm_sdk_core/share/amd_smi
  FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

Exits non-zero if any check fails.
"""

import sys


def check(name, fn):
    try:
        result = fn()
        print(f"[ok] {name}: {result}")
        return True
    except Exception as e:  # noqa: BLE001 - report and fail
        print(f"[FAIL] {name}: {e}")
        return False


ok = True

ok &= check("vllm", lambda: __import__("vllm").__version__)
ok &= check("torch", lambda: __import__("torch").__version__)
ok &= check("flash-attn", lambda: __import__("flash_attn").__version__)


def hip():
    import torch
    return f"HIP available: {torch.cuda.is_available()}"


ok &= check("hip", hip)

sys.exit(0 if ok else 1)
