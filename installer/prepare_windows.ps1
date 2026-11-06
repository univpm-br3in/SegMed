# Run this on Windows before compiling segmed_cpu.iss.
$ErrorActionPreference = "Stop"
$uvVersion = "0.10.0"
$root = $PSScriptRoot
$build = Join-Path $root "build"
$bin = Join-Path $root "bin"
New-Item -ItemType Directory -Force -Path $build, $bin | Out-Null

$archive = Join-Path $build "uv.zip"
$extracted = Join-Path $build "uv-unpacked"
$uri = "https://github.com/astral-sh/uv/releases/download/$uvVersion/uv-x86_64-pc-windows-msvc.zip"
Write-Host "Downloading official uv $uvVersion ($uri)"
Invoke-WebRequest -Uri $uri -OutFile $archive
if (Test-Path $extracted) { Remove-Item $extracted -Recurse -Force }
Expand-Archive -Path $archive -DestinationPath $extracted -Force
$exe = Get-ChildItem -Path $extracted -Filter "uv.exe" -Recurse -File | Select-Object -First 1
if ($null -eq $exe) { throw "uv.exe not found in the downloaded archive" }
$destination = Join-Path $bin "uv.exe"
Copy-Item -Path $exe.FullName -Destination $destination -Force
& $destination --version
if ($LASTEXITCODE -ne 0) { throw "Bundled uv.exe failed its version check" }
Write-Host "Ready. Open segmed.iss in Inno Setup and select Build > Compile."
