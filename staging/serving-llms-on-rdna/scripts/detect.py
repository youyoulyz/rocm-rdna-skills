#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Detect AMD RDNA GPU hardware via amd-smi.

This is the first script the `serving-llms-on-rdna` skill runs. It is
read-only. It answers two questions:

  1. Is there an AMD RDNA consumer GPU on this machine?
  2. What are its properties (gfx version, VRAM, family)?

It prints a human-readable summary and, with `--json`, emits a structured
record the agent can parse to drive the next steps.

Exit codes:
  0 = RDNA GPU found; the rest of the skill can proceed.
  1 = amd-smi failed or no AMD GPU detected.
  2 = AMD GPU found but not RDNA (e.g. Instinct gfx942) -- route to
      the serving-llms-on-instinct skill instead.
  3 = RDNA GPU found but a hard prerequisite is missing (e.g. amdgpu
      kernel module not loaded).

This script is local-only: consumer desktops have no remote SSH workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

# gfx prefix -> (family name, marketing description)
RDNA_FAMILIES = {
    "gfx103": ("rdna2", "Radeon RX 6000 series"),
    "gfx110": ("rdna3", "Radeon RX 7000 series"),
    "gfx115": ("rdna35", "RDNA 3.5 APUs"),
    "gfx120": ("rdna4", "Radeon RX 9000 series"),
}


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


def _classify_gfx(gfx_version):
    """Return (family, is_rdna). family is None when not RDNA."""
    if not gfx_version:
        return None, False
    gfx = gfx_version.lower()
    for prefix, (family, _desc) in RDNA_FAMILIES.items():
        if gfx.startswith(prefix):
            return family, True
    return None, False


def _probe_gpus():
    """Return (gpus, error). gpus is a list of dicts from amd-smi."""
    rc, out, err = _run("amd-smi static --asic --vram --json")
    if rc != 0 and "required groups" in err:
        # User not in video/render group -- retry with sudo
        rc, out, err = _run("sudo amd-smi static --asic --vram --json")
    if rc != 0:
        return None, err or f"amd-smi failed (exit {rc})"

    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:
        return None, f"amd-smi JSON parse failed: {e}"

    if isinstance(data, list):
        gpu_list = data
    elif isinstance(data, dict):
        gpu_list = data.get("gpu_data", [data])
    else:
        gpu_list = [data]

    gpus = []
    for entry in gpu_list:
        asic = entry.get("asic", {})
        vram = entry.get("vram", {})
        vram_size = vram.get("size", {})
        vram_mb = vram_size.get("value") if isinstance(vram_size, dict) else vram_size
        gpus.append({
            "index": entry.get("gpu", len(gpus)),
            "market_name": asic.get("market_name", "Unknown"),
            "gfx_version": asic.get("target_graphics_version", "unknown").lower(),
            "vram_gb": round(vram_mb / 1024, 1) if vram_mb else None,
            "vram_type": vram.get("type"),
            "compute_units": asic.get("num_compute_units"),
        })
    return gpus, None


def _probe_rocm_version():
    rc, out, err = _run("amd-smi version --json", timeout=10)
    if rc != 0 and "required groups" in err:
        rc, out, _ = _run("sudo amd-smi version --json", timeout=10)
    if rc != 0:
        return "unknown"
    try:
        vdata = json.loads(out)
        if isinstance(vdata, list) and vdata:
            return vdata[0].get("rocm_version", "unknown")
        if isinstance(vdata, dict):
            return vdata.get("rocm_version", "unknown")
    except json.JSONDecodeError:
        pass
    return "unknown"


def _probe_system_ram_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _probe_amdgpu_module():
    rc, out, _ = _run("lsmod 2>/dev/null | grep -c '^amdgpu'")
    return out.strip() != "0"


def _probe_gfx_overrides():
    """Find any HSA_OVERRIDE_GFX_VERSION set in the environment or shell rc."""
    val = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
    if val:
        return val
    return None


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--json", action="store_true", help="Emit JSON instead of human text")
    args = p.parse_args()

    gpus, err = _probe_gpus()

    if gpus is None or not gpus:
        # Fall back: is amdgpu loaded but amd-smi missing?
        amdgpu_loaded = _probe_amdgpu_module()
        result = {
            "gpu_count": 0,
            "gfx_version": "unknown",
            "is_rdna": False,
            "error": err or "No AMD GPU detected",
            "hint": "Is amd-smi installed? Is the amdgpu kernel module loaded? Try: lsmod | grep amdgpu",
        }
        if not amdgpu_loaded:
            result["hint"] = "amdgpu kernel module not loaded. Check BIOS and driver installation."
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"ERROR: {result['error']}\n{result['hint']}")
        sys.exit(1)

    rocm_version = _probe_rocm_version()
    system_ram = _probe_system_ram_gb()
    gfx_version = gpus[0]["gfx_version"]
    family, is_rdna = _classify_gfx(gfx_version)

    result = {
        "gpu_count": len(gpus),
        "gfx_version": gfx_version,
        "gpu_family": family,
        "is_rdna": is_rdna,
        "rocm_version": rocm_version,
        "system_ram_gb": system_ram,
        "gpus": gpus,
    }

    if not is_rdna:
        result["route_to"] = "serving-llms-on-instinct"
        result["error"] = (
            f"Detected {gfx_version} which is not a RDNA consumer GPU. "
            "This skill targets RDNA (RX 6000/7000/9000 series). "
            "For Instinct GPUs use the serving-llms-on-instinct skill."
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"NOT RDNA: {gfx_version}. Use serving-llms-on-instinct for Instinct GPUs.")
        sys.exit(2)

    # Check gfx support in installed ROCm
    if rocm_version != "unknown":
        major = rocm_version.split(".")[0]
        gfx_req = {
            "gfx103": "6.0",
            "gfx110": "5.7",
            "gfx115": "6.4",
            "gfx120": "7.3",
        }
        family_prefix = gfx_version[:5]
        min_ver = gfx_req.get(family_prefix)
        if min_ver and rocm_version < min_ver:
            result["warning"] = (
                f"ROCm {rocm_version} may not support {gfx_version} "
                f"(needs >= {min_ver}). Consider upgrading."
            )

    # HSA_OVERRIDE_GFX_VERSION check
    override = _probe_gfx_overrides()
    expected_override = _expected_override(gfx_version)
    if override and override != expected_override:
        result["warning"] = (
            f"HSA_OVERRIDE_GFX_VERSION={override} is set but this GPU expects "
            f"{expected_override}. This can cause kernel compilation failures."
        )

    result["expected_hsa_override"] = expected_override

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        g = gpus[0]
        print(f"GPU: {g['market_name']} ({gfx_version}, {family})")
        print(f"VRAM: {g['vram_gb']} GB {g.get('vram_type') or ''}".rstrip())
        print(f"CUs: {g['compute_units']}  ROCm: {rocm_version}")
        print(f"System RAM: {system_ram} GB")
        print(f"HSA_OVERRIDE_GFX_VERSION: {expected_override}")
    sys.exit(0)


def _expected_override(gfx_version):
    """Expected HSA_OVERRIDE_GFX_VERSION for a gfx version."""
    if gfx_version.startswith("gfx110"):
        return "11.0.0"
    if gfx_version.startswith("gfx103"):
        return "10.3.0"
    if gfx_version.startswith("gfx115"):
        return "11.5.0"
    if gfx_version.startswith("gfx120"):
        return "12.0.0"
    return None


if __name__ == "__main__":
    main()
