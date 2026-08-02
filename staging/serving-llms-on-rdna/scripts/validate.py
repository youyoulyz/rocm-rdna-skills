#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Validate the environment on an AMD RDNA GPU machine before serving LLMs.

This is the second script the `serving-llms-on-rdna` skill runs. It is
read-only (except --auto-fix which applies safe fixes). It checks the
prerequisites for all four backends and reports which are available:

  - vllm-pip          vLLM installed in a conda env with ROCm PyTorch
  - vllm-docker       vLLM ROCm Docker image available
  - llama-cpp-docker  llama.cpp ROCm Docker image available
  - llama-cpp-compile Build tools (cmake, hipcc, clang) for source build

Checks are classified: error (blocks that backend), warning (degrades
perf), advisory (info only).

Usage:
    python scripts/validate.py
    python scripts/validate.py --backend vllm-pip
    python scripts/validate.py --auto-fix

Exit codes: 0 = requested backend ready, 1 = not ready, 2 = bad args.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

# Preferred conda envs for the vllm-pip backend, in order. rocm7.14 is
# the TheRock ROCm wheel stack (Python 3.14); "torch" is the legacy env.
PREFERRED_ENVS = ["rocm7.14", "torch"]


def _detect_conda_env():
    """Find the first usable conda env, or return the first in the list."""
    rc, out, _ = _run("conda env list 2>/dev/null | awk '{print $1}'")
    available = set(out.split()) if rc == 0 else set()
    for name in PREFERRED_ENVS:
        if name in available:
            return name
    return PREFERRED_ENVS[0]


def _run(cmd, timeout=30):
    try:
        r = subprocess.run(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout,
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", f"Command timed out after {timeout}s"


def _have(cmd):
    return shutil.which(cmd) is not None


def check_dev_kfd(issues):
    rc, out, _ = _run("test -e /dev/kfd && echo exists || echo missing")
    if "missing" in out:
        issues.append({
            "check": "dev_kfd", "severity": "error",
            "message": "/dev/kfd not found. amdgpu kernel module not loaded or driver not installed.",
            "fix": "sudo modprobe amdgpu  # or install ROCm driver",
        })
    else:
        rc2, out2, _ = _run("test -r /dev/kfd && echo ok || echo denied")
        if "denied" in out2:
            issues.append({
                "check": "dev_kfd", "severity": "warning",
                "message": "/dev/kfd exists but current user is not in video/render group.",
                "fix": "sudo usermod -aG video,render $USER  # then re-login",
            })


def check_dev_dri(issues):
    rc, out, _ = _run("ls /dev/dri/renderD* 2>/dev/null | wc -l")
    try:
        count = int(out)
    except ValueError:
        count = 0
    if count == 0:
        issues.append({
            "check": "dev_dri", "severity": "error",
            "message": "No /dev/dri/renderD* nodes found. GPU render nodes not present.",
            "fix": "Check amdgpu driver: lsmod | grep amdgpu",
        })


def check_rocm(issues):
    rc, out, _ = _run("cat /opt/rocm/.info/version 2>/dev/null")
    if rc == 0 and out:
        issues.append({
            "check": "rocm_version", "severity": "advisory",
            "message": f"ROCm {out} installed at /opt/rocm.",
            "fix": "",
        })
    else:
        issues.append({
            "check": "rocm_version", "severity": "warning",
            "message": "ROCm not detected at /opt/rocm/.info/version.",
            "fix": "Install ROCm: https://rocm.docs.amd.com",
        })


def check_hsa_override(issues):
    val = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
    if val:
        issues.append({
            "check": "hsa_override", "severity": "advisory",
            "message": f"HSA_OVERRIDE_GFX_VERSION={val} is set in the environment.",
            "fix": "Ensure it matches your GPU. For gfx1100 use 11.0.0.",
        })
    else:
        issues.append({
            "check": "hsa_override", "severity": "warning",
            "message": "HSA_OVERRIDE_GFX_VERSION not set. RDNA consumer GPUs need it for ROCm compute.",
            "fix": "export HSA_OVERRIDE_GFX_VERSION=11.0.0  # for gfx1100",
        })


def check_conda(issues, backends):
    if "vllm-pip" not in backends and "llama-cpp-compile" not in backends:
        return
    if _have("conda") or _have("mamba"):
        issues.append({
            "check": "conda", "severity": "advisory",
            "message": "Conda available.",
            "fix": "",
        })
    else:
        issues.append({
            "check": "conda", "severity": "advisory",
            "message": "Conda not found on PATH. vllm-pip and llama-cpp-compile prefer a conda env.",
            "fix": "Install miniconda: https://docs.conda.io/en/latest/miniconda.html",
        })


def check_conda_env(issues, backends):
    if "vllm-pip" not in backends:
        return
    rc, out, _ = _run(f"conda env list 2>/dev/null | grep -E '^{CONDA_ENV}\\s'")
    if rc != 0 or not out:
        issues.append({
            "check": "conda_env", "severity": "warning",
            "message": f"Conda env '{CONDA_ENV}' not found. vllm-pip backend needs it.",
            "fix": f"conda create -n {CONDA_ENV} python=3.10",
        })
    else:
        issues.append({
            "check": "conda_env", "severity": "advisory",
            "message": f"Conda env '{CONDA_ENV}' exists.",
            "fix": "",
        })


def check_vllm_pip(issues):
    """Check vLLM in the conda env via conda run.

    Two separate signals:
      1. vLLM imports at all (version printed)
      2. ABI sanity: the wheel must match the env's Python version.
         ROCm vLLM wheels on wheels.vllm.ai are cp312-only; a cp312
         wheel force-installed into a cp310 env imports but crashes at
         runtime (`'_C' object has no attribute 'rms_norm'`).
    """
    rc, out, err = _run(
        f"conda run -n {CONDA_ENV} python -c 'import vllm; print(vllm.__version__)' 2>&1",
        timeout=60,
    )
    if rc != 0 or not out:
        issues.append({
            "check": "vllm_pip", "severity": "warning",
            "message": f"vLLM not importable in conda env '{CONDA_ENV}'. Output: {(out or err)[:150]}",
            "fix": f"conda create -n vllm python=3.12 && conda activate vllm && pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/",
        })
        return

    issues.append({
        "check": "vllm_pip", "severity": "advisory",
        "message": f"vLLM {out} installed in conda env '{CONDA_ENV}'.",
        "fix": "",
    })

    # ABI check: Python version of the env vs the wheel the vLLM was built
    # for. The dist-info dir name carries no cp tag; the wheel filename
    # (with cpXY) is recorded in direct_url.json inside dist-info.
    rc_py, py_ver, _ = _run(
        f"conda run -n {CONDA_ENV} python -c 'import sys; print(sys.version_info[:2])' 2>&1",
        timeout=30,
    )
    rc_du, direct_url, _ = _run(
        f"conda run -n {CONDA_ENV} python -c "
        "'import glob, json; f = glob.glob(__import__(\"vllm\").__file__.rsplit(\"/\",1)[0] + \"/../vllm-*.dist-info/direct_url.json\")[0]; print(json.load(open(f))[\"url\"])' 2>&1",
        timeout=30,
    )
    import re
    # cpXYZ in wheel filenames: cp312 = Python 3.12
    m = re.search(r"cp(\d)(\d{2})", direct_url) if direct_url else None
    if rc_py == 0 and rc_du == 0 and m:
        wheel_py = (int(m.group(1)), int(m.group(2)))
        env_py = tuple(int(x) for x in py_ver.strip("()").split(",")[:2])
        if wheel_py != env_py:
            issues.append({
                "check": "vllm_abi", "severity": "error",
                "message": (
                    f"vLLM wheel is built for Python {wheel_py[0]}.{wheel_py[1]} "
                    f"(from direct_url.json) but the env '{CONDA_ENV}' runs "
                    f"Python {env_py[0]}.{env_py[1]}. The C extension will fail "
                    "at runtime ('_C' object has no attribute 'rms_norm')."
                ),
                "fix": "Create a Python 3.12 env: conda create -n vllm python=3.12 && conda activate vllm && pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/",
            })
        else:
            issues.append({
                "check": "vllm_abi", "severity": "advisory",
                "message": f"vLLM wheel Python ({wheel_py[0]}.{wheel_py[1]}) matches env Python. OK.",
                "fix": "",
            })


def check_rocm_env_vars(issues):
    """Check the mandatory exports for the TheRock ROCm wheel stack."""
    pyp = os.environ.get("PYTHONPATH", "")
    flash = os.environ.get("FLASH_ATTENTION_TRITON_AMD_ENABLE", "")
    if "amd_smi" not in pyp:
        issues.append({
            "check": "rocm_pythonpath", "severity": "error",
            "message": (
                "PYTHONPATH does not include the amd_smi bindings. The TheRock "
                "ROCm wheel stack needs: "
                "export PYTHONPATH=$CONDA_PREFIX/lib/python3.14/site-packages/_rocm_sdk_core/share/amd_smi"
            ),
            "fix": "export PYTHONPATH=$CONDA_PREFIX/lib/python3.14/site-packages/_rocm_sdk_core/share/amd_smi",
        })
    if flash != "TRUE":
        issues.append({
            "check": "flash_triton", "severity": "error",
            "message": "FLASH_ATTENTION_TRITON_AMD_ENABLE is not TRUE. flash-attn is unavailable on RDNA without it.",
            "fix": "export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE",
        })


def check_torch_rocm(issues):
    """Check PyTorch sees the GPU in the conda env."""
    rc, out, err = _run(
        f"conda run -n {CONDA_ENV} python -c "
        "'import torch; print(torch.cuda.is_available())' 2>&1",
        timeout=60,
    )
    if rc == 0 and "True" in out:
        issues.append({
            "check": "torch_rocm", "severity": "advisory",
            "message": f"PyTorch sees the GPU: {out}",
            "fix": "",
        })
    else:
        issues.append({
            "check": "torch_rocm", "severity": "error",
            "message": f"PyTorch in conda env '{CONDA_ENV}' does not see the GPU. Output: {out or err[:200]}",
            "fix": f"conda run -n {CONDA_ENV} pip install torch --index-url https://download.pytorch.org/whl/rocm7.2",
        })


def check_docker(issues, backends):
    if "vllm-docker" not in backends and "llama-cpp-docker" not in backends:
        return
    rc, out, err = _run("docker ps -q 2>&1 | head -1")
    if rc != 0 or "permission denied" in err.lower() or "cannot connect" in err.lower():
        issues.append({
            "check": "docker", "severity": "error",
            "message": f"Docker not accessible: {err or 'docker ps failed'}",
            "fix": "sudo systemctl start docker  |  sudo usermod -aG docker $USER",
        })
    else:
        issues.append({
            "check": "docker", "severity": "advisory",
            "message": "Docker accessible.",
            "fix": "",
        })


def check_docker_gpu(issues, backends):
    """Verify GPU passthrough with a one-shot container."""
    if "vllm-docker" not in backends and "llama-cpp-docker" not in backends:
        return
    # First check if the probe image is present; if not, skip (first pull
    # would take minutes and the message is just "image not pulled yet").
    rc_im, out_im, _ = _run("docker images rocm/dev-ubuntu-22.04 --format '{{.Tag}}' 2>/dev/null | head -1")
    if not out_im.strip():
        issues.append({
            "check": "docker_gpu", "severity": "advisory",
            "message": "Docker GPU passthrough not yet verified — probe image rocm/dev-ubuntu-22.04 not pulled. It will be verified on first container launch.",
            "fix": "docker pull rocm/dev-ubuntu-22.04:latest  # optional pre-verification",
        })
        return
    rc, out, err = _run(
        "docker run --rm --device /dev/kfd --device /dev/dri "
        "--group-add=video --security-opt seccomp=unconfined "
        "rocm/dev-ubuntu-22.04:latest rocminfo 2>&1 | head -5",
        timeout=120,
    )
    if rc == 0 and ("Agent 1" in out or "GPU" in out):
        issues.append({
            "check": "docker_gpu", "severity": "advisory",
            "message": "Docker GPU passthrough verified (rocminfo sees the GPU).",
            "fix": "",
        })
    else:
        issues.append({
            "check": "docker_gpu", "severity": "warning",
            "message": f"Docker GPU passthrough check failed: {err or out[:200]}",
            "fix": "Ensure /dev/kfd and /dev/dri are passed with --device and user is in video group.",
        })


def check_docker_image(issues, backend, image):
    if backend not in ("vllm-docker", "llama-cpp-docker"):
        return
    name = image.split(":")[0]
    rc, out, _ = _run(f"docker images {name} --format '{{{{.Tag}}}}' 2>/dev/null | head -1")
    if not out.strip():
        issues.append({
            "check": f"{backend}_image", "severity": "advisory",
            "message": f"Docker image {image} not pulled yet. First launch downloads it.",
            "fix": f"docker pull {image}",
        })
    else:
        issues.append({
            "check": f"{backend}_image", "severity": "advisory",
            "message": f"Docker image {image} pulled (tag {out}).",
            "fix": "",
        })


def check_build_tools(issues, backends):
    if "llama-cpp-compile" not in backends:
        return
    if not _have("cmake"):
        issues.append({
            "check": "build_cmake", "severity": "error",
            "message": "cmake not found. Needed for llama.cpp source build.",
            "fix": "sudo apt install cmake",
        })
    if not _have("hipcc"):
        issues.append({
            "check": "build_hipcc", "severity": "error",
            "message": "hipcc not found. ROCm must be installed.",
            "fix": "Install ROCm or add /opt/rocm/bin to PATH.",
        })
    if _have("hipcc"):
        rc, out, _ = _run("hipcc --version 2>&1 | head -1")
        issues.append({
            "check": "build_hipcc", "severity": "advisory",
            "message": f"hipcc: {out}",
            "fix": "",
        })
    if not _have("make") and not _have("ninja"):
        issues.append({
            "check": "build_make", "severity": "warning",
            "message": "Neither make nor ninja found. CMake needs a build generator.",
            "fix": "sudo apt install build-essential",
        })
    if not _have("git"):
        issues.append({
            "check": "build_git", "severity": "error",
            "message": "git not found. Needed to clone llama.cpp.",
            "fix": "sudo apt install git",
        })
    if _have("hipcc"):
        rc, clang, _ = _run("hipconfig -l 2>/dev/null")
        if rc == 0 and clang:
            rc2, _, _ = _run(f"test -x {clang}/clang && echo yes || echo no")
            if rc2 == 0:
                issues.append({
                    "check": "build_clang", "severity": "advisory",
                    "message": f"ROCm clang at {clang}/clang.",
                    "fix": "",
                })
            else:
                issues.append({
                    "check": "build_clang", "severity": "error",
                    "message": f"ROCm clang not found at {clang}/clang.",
                    "fix": "Reinstall ROCm with full toolchain.",
                })


def check_disk(issues):
    rc, out, _ = _run("df -BG ~/.cache/huggingface 2>/dev/null | tail -1 | awk '{print $4}'")
    if rc == 0 and out:
        free_gb = int(out.rstrip("G"))
        if free_gb < 50:
            issues.append({
                "check": "disk_space", "severity": "warning",
                "message": f"Only ~{free_gb} GB free in ~/.cache/huggingface. Models need 10-20 GB each.",
                "fix": "Free disk space or mount a larger volume.",
            })
    else:
        issues.append({
            "check": "disk_space", "severity": "advisory",
            "message": "~/.cache/huggingface does not exist yet (created on first model download).",
            "fix": "",
        })


def check_cuda_visible(issues):
    val = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if val == "" and "CUDA_VISIBLE_DEVICES" in os.environ:
        issues.append({
            "check": "cuda_visible_devices", "severity": "error",
            "message": "CUDA_VISIBLE_DEVICES is set to an empty string. This hides all GPUs from the ROCm runtime.",
            "fix": "unset CUDA_VISIBLE_DEVICES",
        })
    elif val:
        issues.append({
            "check": "cuda_visible_devices", "severity": "advisory",
            "message": f"CUDA_VISIBLE_DEVICES={val}. ROCm maps this to HIP_VISIBLE_DEVICES.",
            "fix": "",
        })


def check_hf_token(issues):
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        issues.append({
            "check": "hf_token", "severity": "advisory",
            "message": "HF_TOKEN not set. Required for gated models (Llama, Gemma). Not needed for Qwen.",
            "fix": "export HF_TOKEN=hf_...",
        })


def check_ollama(issues):
    if _have("ollama"):
        issues.append({
            "check": "ollama", "severity": "advisory",
            "message": "Ollama available as alternative quick path.",
            "fix": "",
        })


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--backend", default="auto",
                   choices=["auto", "vllm-pip", "vllm-docker", "llama-cpp-docker", "llama-cpp-compile"],
                   help="Which backend to validate (default: all)")
    p.add_argument("--conda-env", default="",
                   help="Conda env to check for vllm-pip (default: rocm7.14, then torch)")
    p.add_argument("--auto-fix", action="store_true", help="Apply safe fixes without prompting")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of human text")
    args = p.parse_args()

    global CONDA_ENV
    CONDA_ENV = args.conda_env or _detect_conda_env()

    if args.backend == "auto":
        backends = ["vllm-pip", "vllm-docker", "llama-cpp-docker", "llama-cpp-compile"]
    else:
        backends = [args.backend]

    issues = []

    # Common checks (all backends)
    check_dev_kfd(issues)
    check_dev_dri(issues)
    check_rocm(issues)
    check_hsa_override(issues)
    check_cuda_visible(issues)
    check_hf_token(issues)
    check_disk(issues)
    check_ollama(issues)

    # Backend-specific
    check_conda(issues, backends)
    check_conda_env(issues, backends)
    if "vllm-pip" in backends:
        check_rocm_env_vars(issues)
        check_vllm_pip(issues)
        check_torch_rocm(issues)
    check_docker(issues, backends)
    check_docker_gpu(issues, backends)
    check_docker_image(issues, "vllm-docker", "rocm/vllm-dev:navi_nightly")
    check_docker_image(issues, "llama-cpp-docker", "ghcr.io/ggml-org/llama.cpp:full-rocm")
    check_build_tools(issues, backends)

    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    advisories = [i for i in issues if i["severity"] == "advisory"]

    # Per-backend readiness (errors specific to a backend block it; common errors block all)
    common_blocks = {"dev_kfd", "dev_dri", "cuda_visible_devices"}
    backend_map = {
        "vllm-pip": {"conda_env", "vllm_pip", "torch_rocm", "vllm_abi",
                     "rocm_pythonpath", "flash_triton"},
        "vllm-docker": {"docker", "docker_gpu", "vllm-docker_image"},
        "llama-cpp-docker": {"docker", "docker_gpu", "llama-cpp-docker_image"},
        "llama-cpp-compile": {"build_cmake", "build_hipcc", "build_git", "build_clang"},
    }
    backend_ready = {}
    for b in backends:
        blocking = common_blocks | backend_map.get(b, set())
        backend_ready[b] = not any(e["check"] in blocking for e in errors)

    ready = backend_ready.get(args.backend, all(backend_ready.values())) if args.backend != "auto" else any(backend_ready.values())

    result = {
        "ready": ready,
        "backends_available": backend_ready,
        "conda_env": CONDA_ENV,
        "errors": errors,
        "warnings": warnings,
        "advisories": advisories,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=== Errors ===")
        for e in errors:
            print(f"  [ERROR] {e['message']}")
            if e.get("fix"):
                print(f"          Fix: {e['fix']}")
        print("=== Warnings ===")
        for w in warnings:
            print(f"  [WARN] {w['message']}")
        print("=== Advisories ===")
        for a in advisories:
            print(f"  [INFO] {a['message']}")
        print("=== Backends ===")
        for b, r in backend_ready.items():
            print(f"  {b}: {'READY' if r else 'NOT READY'}")

    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
