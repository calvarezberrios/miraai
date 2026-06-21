<#
  01-install-prereqs.ps1  --  RUN AS ADMINISTRATOR
  Installs the Windows prerequisites for GPU Docker:
    - WSL2 (kernel + VirtualMachinePlatform + WSL feature)
    - Docker Desktop (WSL2 backend, which exposes --gpus all via the host NVIDIA driver)

  Hardware already verified on this box:
    GPU  : GTX 1660 SUPER, 6 GB VRAM, driver 595.97 (>=470 -> WSL2 CUDA OK)
    RAM  : 24 GB     Virtualization: ENABLED (VBS running)

  A REBOOT is required after WSL features are enabled. Re-run is safe (idempotent).
#>

# --- self-elevate ---------------------------------------------------------
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
          ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Not elevated -- relaunching as Administrator (approve the UAC prompt)..." -ForegroundColor Yellow
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    return
}

$ErrorActionPreference = 'Stop'
Write-Host "== Step 1/3: enable WSL + VirtualMachinePlatform features ==" -ForegroundColor Cyan
$f1 = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
$f2 = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
$needReboot = $false
if ($f1.State -ne 'Enabled') {
    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -NoRestart | Out-Null
    $needReboot = $true
}
if ($f2.State -ne 'Enabled') {
    Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -NoRestart | Out-Null
    $needReboot = $true
}
Write-Host "  WSL feature: $($f1.State)   VirtualMachinePlatform: $($f2.State)"

Write-Host "== Step 2/3: install/update WSL2 kernel ==" -ForegroundColor Cyan
# --no-distribution keeps it minimal; Docker Desktop ships its own docker-desktop distro.
try { wsl --update } catch { Write-Host "  wsl --update skipped: $($_.Exception.Message)" }
try { wsl --set-default-version 2 } catch {}

Write-Host "== Step 3/3: install Docker Desktop (winget) ==" -ForegroundColor Cyan
$docker = Get-Command docker -ErrorAction SilentlyContinue
if (-not $docker) {
    winget install -e --id Docker.DockerDesktop --accept-source-agreements --accept-package-agreements
} else {
    Write-Host "  docker already present: $($docker.Source)"
}

Write-Host ""
if ($needReboot) {
    Write-Host "REBOOT REQUIRED to finish enabling WSL2." -ForegroundColor Red
    Write-Host "After reboot: launch 'Docker Desktop' once, wait for the whale icon to go steady," -ForegroundColor Yellow
    Write-Host "then run  02-download-model.ps1  (no admin needed)." -ForegroundColor Yellow
    $r = Read-Host "Reboot now? (y/N)"
    if ($r -eq 'y') { Restart-Computer -Force }
} else {
    Write-Host "Prereqs look good. Start Docker Desktop, then run 02-download-model.ps1." -ForegroundColor Green
}
