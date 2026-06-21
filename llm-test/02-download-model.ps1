<#
  02-download-model.ps1  --  no admin required
  Downloads the GGUF into D:\models (462 GB free; kept OUT of the git repo).
  Resumable: if the connection drops, just re-run -- curl -C - continues.

  Model: bartowski/Qwen_Qwen3.6-35B-A3B-GGUF  ->  Q4_K_M  (20.75 GB, single file)
#>
param(
    [string]$ModelDir = "D:\models",
    [string]$File     = "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"
)

$repo = "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF"
$url  = "https://huggingface.co/$repo/resolve/main/$File`?download=true"
$dest = Join-Path $ModelDir $File

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

# free-space guard (need ~21 GB)
$drive = (Get-Item $ModelDir).PSDrive
$freeGB = [math]::Round((Get-PSDrive $drive.Name).Free / 1GB, 1)
Write-Host "Target: $dest"
Write-Host "Free on $($drive.Name): $freeGB GB"
if ($freeGB -lt 25) { Write-Warning "Less than 25 GB free -- the file is ~21 GB. Aborting."; return }

Write-Host "Downloading (resumable). This is ~21 GB; grab a coffee..." -ForegroundColor Cyan
# curl.exe ships with Windows 10+. -C - resumes a partial file; -L follows the CDN redirect.
curl.exe -L -C - -o "$dest" "$url"

if (Test-Path $dest) {
    $sz = [math]::Round((Get-Item $dest).Length / 1GB, 2)
    Write-Host "Done: $dest  ($sz GB)" -ForegroundColor Green
    if ($sz -lt 20) { Write-Warning "File smaller than expected (~20.75 GB) -- download may be incomplete; re-run to resume." }
} else {
    Write-Warning "Download did not produce a file."
}
