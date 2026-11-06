"""Select supported PyTorch wheels using the NVIDIA driver and GPU architecture.

A candidate is only accepted after the installer runs a CUDA computation.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


DRIVER_CU126 = (560, 76)
DRIVER_CU128 = (570, 65)
MIN_COMPUTE_CAPABILITY = (5, 0)
CU128_MIN_COMPUTE_CAPABILITY = (7, 5)
BLACKWELL_MIN_COMPUTE_CAPABILITY = (10, 0)


def _version(value):
    match = re.match(r"^(\d+)\.(\d+)", value.strip())
    if not match:
        raise ValueError(f"Invalid version: {value!r}")
    return int(match.group(1)), int(match.group(2))


def _nvidia_smi():
    executable = shutil.which("nvidia-smi.exe") or shutil.which("nvidia-smi")
    if executable:
        return executable

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    windir = os.environ.get("WINDIR", r"C:\Windows")
    for path in (
        Path(program_files) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
        Path(windir) / "System32" / "nvidia-smi.exe",
    ):
        if path.is_file():
            return str(path)
    return None


def parse_gpu_info(output):
    """Parse nvidia-smi CSV: index,name,uuid,driver_version,compute_cap."""
    gpus = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 5:
            raise ValueError(f"Unexpected NVIDIA GPU information: {line!r}")
        # Take four fields from the right in case the product name contains a comma.
        name = ",".join(fields[1:-3]).strip()
        index = int(fields[0])
        gpu_uuid, driver_text, compute_text = fields[-3:]
        if not gpu_uuid.startswith("GPU-"):
            raise ValueError(f"Unexpected NVIDIA GPU UUID: {gpu_uuid!r}")
        driver = _version(driver_text)
        capability = _version(compute_text)
        gpus.append({
            "index": index,
            "name": name,
            "uuid": gpu_uuid,
            "driver": driver_text,
            "compute_capability": compute_text,
            "driver_tuple": driver,
            "compute_tuple": capability,
        })
    return gpus


def get_gpu_info(sample_file=None):
    if sample_file:
        return parse_gpu_info(Path(sample_file).read_text(encoding="utf-8")), None

    executable = _nvidia_smi()
    if not executable:
        return [], "nvidia-smi was not found; no supported NVIDIA driver detected"

    try:
        result = subprocess.run(
            [executable, "--query-gpu=index,name,uuid,driver_version,compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"Unable to query NVIDIA hardware: {exc}"
    if result.returncode != 0:
        return [], f"NVIDIA query failed: {result.stderr.strip() or result.stdout.strip()}"
    try:
        return parse_gpu_info(result.stdout), None
    except ValueError as exc:
        return [], str(exc)


def candidates_for_gpu(gpu):
    driver = gpu["driver_tuple"]
    compute = gpu["compute_tuple"]
    if compute < MIN_COMPUTE_CAPABILITY:
        return [], "GPU compute capability below 5.0 is outside this PyTorch 2.8 support profile"
    if compute >= BLACKWELL_MIN_COMPUTE_CAPABILITY:
        if driver >= DRIVER_CU128:
            return ["CUDA128"], None
        return [], "Blackwell-class GPU requires the CUDA 12.8 build and a newer NVIDIA driver"
    if compute < CU128_MIN_COMPUTE_CAPABILITY:
        if driver >= DRIVER_CU126:
            return ["CUDA126"], None
        return [], "Older GPU requires CUDA 12.6; the NVIDIA driver is below the tested baseline"
    results = []
    if driver >= DRIVER_CU128:
        results.append("CUDA128")
    if driver >= DRIVER_CU126:
        results.append("CUDA126")
    if results:
        return results, None
    return [], "NVIDIA driver is below the tested CUDA 12.6 baseline"


def select(gpus, error=None):
    candidates = []
    diagnostics = []
    for gpu in gpus:
        options, reason = candidates_for_gpu(gpu)
        diagnostics.append({
            "index": gpu["index"],
            "name": gpu["name"],
            "driver": gpu["driver"],
            "compute_capability": gpu["compute_capability"],
            "supported_backends": options,
            "reason": reason,
        })
        for backend in options:
            candidates.append({
                "backend": backend,
                "gpu_index": gpu["index"],
                "gpu_name": gpu["name"],
                "gpu_uuid": gpu["uuid"],
                "driver": gpu["driver"],
                "compute_capability": gpu["compute_capability"],
            })

    # Prefer CUDA 12.8 when usable, then CUDA 12.6. For identical builds,
    # use the GPU with the higher compute capability, then the lower index.
    candidates.sort(key=lambda candidate: (
        1 if candidate["backend"] == "CUDA128" else 0,
        _version(candidate["compute_capability"]),
        -candidate["gpu_index"],
    ), reverse=True)

    return {
        "candidates": candidates,
        "gpus": diagnostics,
        "message": error or ("No NVIDIA GPU detected" if not gpus else ""),
        "cpu_fallback": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect NVIDIA GPU support for SegMed's PyTorch 2.8.0 builds")
    parser.add_argument("--sample-file", help="Use recorded nvidia-smi output (for validation only)")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    gpus, error = get_gpu_info(args.sample_file)
    print(json.dumps(select(gpus, error), indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
