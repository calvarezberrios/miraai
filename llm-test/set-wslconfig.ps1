<#
  set-wslconfig.ps1  --  give the WSL2 VM enough RAM to --mlock a 20.75 GB model.
  By default WSL2 caps at ~50% of host RAM; on a 24 GB box that's ~12 GB -- not
  enough to pin a 20.75 GB model. This writes %UserProfile%\.wslconfig.

  NOTE: handing 22 GB to WSL leaves Windows ~2 GB. Close other apps before running
  the model. If Windows thrashes, drop memory to 20GB and lower the context size.
  After running this, apply it with:  wsl --shutdown   (then restart Docker Desktop)
#>
param([int]$MemoryGB = 22, [int]$SwapGB = 8)

$path = Join-Path $env:USERPROFILE ".wslconfig"
$content = @"
[wsl2]
memory=${MemoryGB}GB
swap=${SwapGB}GB
processors=12
"@
Set-Content -Path $path -Value $content -Encoding ascii
Write-Host "Wrote $path :" -ForegroundColor Green
Get-Content $path
Write-Host "`nApply with:  wsl --shutdown   then relaunch Docker Desktop." -ForegroundColor Yellow
