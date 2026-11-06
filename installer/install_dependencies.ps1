param(
    [string]$ProjectPath = $PSScriptRoot,
    [ValidateSet("CPU", "Auto", "CUDA126", "CUDA128", "Preserve")]
    [string]$Backend = "Auto"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "Continue"
$ProjectPath = [System.IO.Path]::GetFullPath($ProjectPath)
$uv = Join-Path $ProjectPath "tools\uv.exe"
$python = Join-Path $ProjectPath ".venv\Scripts\python.exe"
$verification = Join-Path $ProjectPath "verify_environment.py"
$backendCheck = Join-Path $ProjectPath "verify_torch_backend.py"
$selector = Join-Path $ProjectPath "select_torch_backend.py"
$marker = Join-Path $ProjectPath ".installation-complete"
$backendFile = Join-Path $ProjectPath ".inference-backend"
$deviceFile = Join-Path $ProjectPath ".cuda-device-uuid"
$logDir = Join-Path $env:LOCALAPPDATA "SegMed\logs"
New-Item -Path $logDir -ItemType Directory -Force | Out-Null
$logPath = Join-Path $logDir "setup.log"
$transcriptStarted = $false

function Invoke-Checked([string]$Step, [scriptblock]$Operation) {
    Write-Host $Step
    & $Operation
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed (exit $LASTEXITCODE)"
    }
}

function Install-Torch([string]$Kind) {
    switch ($Kind) {
        "CUDA128" {
            $tag = "cu128"
            $torch = "torch==2.8.0+cu128"
            $vision = "torchvision==0.23.0+cu128"
        }
        "CUDA126" {
            $tag = "cu126"
            $torch = "torch==2.8.0+cu126"
            $vision = "torchvision==0.23.0+cu126"
        }
        "CPU" {
            $tag = "cpu"
            $torch = "torch==2.8.0+cpu"
            $vision = "torchvision==0.23.0+cpu"
        }
        default { throw "Invalid PyTorch backend: $Kind" }
    }

    $index = "https://download.pytorch.org/whl/$tag"
    Write-Host "Installing $torch and $vision from $index"
    # The project lockfile installs the other dependencies first.
    & $uv pip install --python $python --no-deps --index-url $index $torch $vision
    if ($LASTEXITCODE -ne 0) { throw "Installing $Kind PyTorch failed (exit $LASTEXITCODE)" }
}

function Test-Torch([string]$Kind) {
    & $python $backendCheck --backend $Kind
    if ($LASTEXITCODE -ne 0) { throw "$Kind PyTorch verification failed" }
}

function Get-GpuCandidates {
    if (-not (Test-Path $selector)) { throw "GPU selector not found: $selector" }
    $output = & $python $selector
    if ($LASTEXITCODE -ne 0) { throw "NVIDIA GPU detection failed" }
    return ($output | Out-String | ConvertFrom-Json)
}

try {
    Start-Transcript -Path $logPath -Append | Out-Null
    $transcriptStarted = $true

    if (-not (Test-Path $uv)) { throw "Bundled uv.exe not found: $uv" }
    if (-not (Test-Path (Join-Path $ProjectPath "uv.lock"))) { throw "Project lockfile not found" }

    $preferredUuid = $null
    if ($Backend -eq "Preserve") {
        if (Test-Path $backendFile) {
            $Backend = (Get-Content -Path $backendFile -Raw).Trim()
            if (Test-Path $deviceFile) {
                $preferredUuid = (Get-Content -Path $deviceFile -Raw).Trim()
            }
        }
        else { $Backend = "Auto" }
        if ($Backend -notin @("CPU", "CUDA126", "CUDA128", "Auto")) {
            throw "Invalid saved backend: $Backend"
        }
    }

    if (Test-Path $marker) { Remove-Item -Path $marker -Force }
    $env:UV_PYTHON_PREFERENCE = "only-managed"
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectPath "runtime\python"
    $env:UV_CACHE_DIR = Join-Path $env:LOCALAPPDATA "SegMed\uv-cache"
    $env:UV_LINK_MODE = "copy"
    Set-Location -Path $ProjectPath

    Invoke-Checked "[1/5] Installing private Python 3.11..." {
        & $uv python install 3.11
    }

    # These two packages are installed from the selected official PyTorch index.
    Invoke-Checked "[2/5] Installing locked SegMed dependencies..." {
        & $uv sync --locked --no-dev --python 3.11 --no-install-package torch --no-install-package torchvision
    }
    if (-not (Test-Path $python)) { throw "Project Python environment was not created" }

    $selected = $null
    $selectedUuid = $null
    $installedKind = $null
    if ($Backend -ne "CPU") {
        $info = Get-GpuCandidates
        if ($info.message) { Write-Host "GPU detection: $($info.message)" }
        foreach ($gpu in @($info.gpus)) {
            $supported = (@($gpu.supported_backends) -join ', ')
            if (-not $supported) { $supported = "none" }
            Write-Host "GPU $($gpu.index): $($gpu.name), driver $($gpu.driver), compute $($gpu.compute_capability), supported: $supported"
            if ($gpu.reason) { Write-Host "  $($gpu.reason)" }
        }

        $candidates = @($info.candidates | Where-Object {
            $Backend -eq "Auto" -or $_.backend -eq $Backend
        })
        if ($preferredUuid) {
            $candidates = @(
                @($candidates | Where-Object { $_.gpu_uuid -eq $preferredUuid }) +
                @($candidates | Where-Object { $_.gpu_uuid -ne $preferredUuid })
            )
        }

        if ($candidates.Count -eq 0) {
            if ($Backend -ne "Auto") {
                throw "No detected NVIDIA GPU supports the requested $Backend configuration. Run the automatic setup or update the NVIDIA driver."
            }
            Write-Host "No GPU matches a supported PyTorch build; selecting CPU."
        }

        foreach ($candidate in $candidates) {
            Write-Host "Trying $($candidate.backend) on $($candidate.gpu_name) (compute $($candidate.compute_capability))."
            # A UUID is unambiguous when multiple NVIDIA GPUs are present.
            $env:CUDA_VISIBLE_DEVICES = $candidate.gpu_uuid
            try {
                if ($installedKind -ne $candidate.backend) {
                    Install-Torch $candidate.backend
                    $installedKind = $candidate.backend
                }
                Test-Torch $candidate.backend
                $selected = $candidate.backend
                $selectedUuid = $candidate.gpu_uuid
                break
            }
            catch {
                Write-Warning "$($candidate.backend) verification failed: $($_.Exception.Message)"
                if ($Backend -ne "Auto") { throw }
            }
        }
    }

    if (-not $selected) {
        $env:CUDA_VISIBLE_DEVICES = $null
        Install-Torch "CPU"
        Test-Torch "CPU"
        $selected = "CPU"
    }

    Invoke-Checked "[4/5] Checking package dependencies..." {
        & $uv pip check --python $python
    }

    Invoke-Checked "[5/5] Verifying SegMed imports..." {
        & $python $verification --backend $selected
    }

    Set-Content -Path $backendFile -Value $selected -Encoding Ascii
    if ($selectedUuid) {
        Set-Content -Path $deviceFile -Value $selectedUuid -Encoding Ascii
    }
    elseif (Test-Path $deviceFile) {
        Remove-Item -Path $deviceFile -Force
    }
    Set-Content -Path $marker -Value "$selected environment configured" -Encoding Ascii
    Write-Host "SegMed setup completed. Selected backend: $selected"
    if ($selectedUuid) { Write-Host "Selected NVIDIA device: $selectedUuid" }
    exit 0
}
catch {
    Write-Host "SegMed setup failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Installation log: $logPath"
    exit 1
}
finally {
    if ($transcriptStarted) { Stop-Transcript | Out-Null }
}
