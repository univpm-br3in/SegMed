# SegMed Windows installer: automatic NVIDIA PyTorch

This project installs SegMed with **uv** inside a private Python 3.11
installation. It does not require a system Python or a separately installed
CUDA Toolkit. PyTorch and TorchVision come from official PyTorch CPU/CUDA
indexes; TensorFlow uses CPU on native Windows.

## Build the installer on Windows

1. Install [Inno Setup](https://jrsoftware.org/isinfo.php), version 6.3 or newer.
2. Open PowerShell in this `installer` directory.
3. Run `powershell -ExecutionPolicy Bypass -File .\prepare_windows.ps1`.
4. Open `segmed.iss` and select **Build > Compile**.
5. Run `output\SegMed-Setup.exe`.

The GPU option is enabled by default, but can be unchecked to install CPU-only
PyTorch. A CPU-only installer (`segmed_cpu.iss`) is also available. Both
installers use the same application ID and folder; do not install both at once.

The installer needs internet access for Python and Python packages. It does not
install or update the NVIDIA driver.

## How automatic selection works

The selector reads GPU **name, UUID, driver version, and compute capability**
using `nvidia-smi`. It supports the following **candidate build profiles**:

- PyTorch `2.8.0+cu128` and TorchVision `0.23.0+cu128`: compute capability
  **7.5–9.x or 10.0+**, with Windows driver **570.65+** (conservative baseline).
- PyTorch `2.8.0+cu126` and TorchVision `0.23.0+cu126`: compute capability
  **5.0–9.x**, with Windows driver **560.76+** (conservative baseline).
- PyTorch `2.8.0+cpu` and TorchVision `0.23.0+cpu`: when there is no supported
  CUDA candidate or GPU tests fail.

For compute capability 10.0+, CUDA 12.6 is intentionally not attempted. For
compute capability 5.0–7.4, CUDA 12.8 is intentionally not attempted. These
are conservative compatibility rules for the **pinned PyTorch 2.8.0 builds**,
not general hardware limits. Some driver/runtime combinations work through
NVIDIA's minor-version compatibility; this initial release does not attempt
them. Older GPU generations or drivers may use CPU until separate legacy
profiles are supported.

For each eligible combination, the installer downloads a matching wheel pair
and verifies a **real GPU matrix multiplication and 3D convolution**. It tries
compatible alternatives before falling back to CPU. It saves the selected GPU
UUID in `.cuda-device-uuid`, and the launcher applies `CUDA_VISIBLE_DEVICES`
for SegMed so the tested GPU is used consistently on multi-GPU machines.

GPU detection and a small computation are **not** a guarantee that every model
will run: inference can still fail from insufficient GPU memory, unsupported
operators, or external model-specific dependencies.

## Test GPU detection without installing anything

If Python 3.11 is already available, run:

```powershell
python .\select_torch_backend.py --pretty
```

This command only inspects the hardware and prints a prioritized JSON list.
It does not install PyTorch. To run the selection tests:

```powershell
python .\test_cuda_selection.py
```

## Check the installed backend

```powershell
& "$env:LOCALAPPDATA\Programs\SegMed\.venv\Scripts\python.exe" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The selected backend (`CPU`, `CUDA126`, or `CUDA128`) is stored in
`.inference-backend` under the SegMed installation folder.

Setup logs: `%LOCALAPPDATA%\SegMed\logs\setup.log`

Startup logs: `%LOCALAPPDATA%\SegMed\logs\launch.log`

## Repair or switch backends

Use the Start menu entries **Repair inference environment** or
**Auto-configure NVIDIA acceleration**. To force a CPU installation:

```powershell
powershell -ExecutionPolicy Bypass -File "$env:LOCALAPPDATA\Programs\SegMed\install_dependencies.ps1" -ProjectPath "$env:LOCALAPPDATA\Programs\SegMed" -Backend CPU
```

The installer uses `uv sync --locked --no-install-package torch
--no-install-package torchvision` for the normal dependencies, then installs
PyTorch from its selected official index with `uv pip install --no-deps`.
Because the original `uv.lock` points to generic PyPI wheels, **do not run a
plain `uv sync` inside the installed SegMed environment**: it may replace the
CUDA wheels. The application shortcut launches its private Python directly,
not via `uv run`.
