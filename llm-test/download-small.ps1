<#
  download-small.ps1  --  no admin required
  Grabs the small DENSE Qwen3 GGUFs into D:\models (kept OUT of the git repo).
  Resumable: curl -C - continues a partial file if the connection drops.

  These are the "leave the box usable" models that run fully on the 6 GB GPU:
    8b  ->  bartowski/Qwen_Qwen3-8B-GGUF   Q4_K_S  (~4.6 GB)  closest to the 35B's feel
    4b  ->  bartowski/Qwen_Qwen3-4B-GGUF   Q4_K_M  (~2.5 GB)  fastest, most headroom

  Usage:
    ./download-small.ps1            # both (default)
    ./download-small.ps1 -Size 8b   # just the 8B
    ./download-small.ps1 -Size 4b   # just the 4B
#>
param(
    [ValidateSet('all','8b','4b')][string]$Size = 'all',
    [string]$ModelDir = "D:\models"
)

# repo + filename per size
$catalog = @{
    '8b' = @{ Repo = 'bartowski/Qwen_Qwen3-8B-GGUF'; File = 'Qwen_Qwen3-8B-Q4_K_S.gguf'; ApproxGB = 4.6 }
    '4b' = @{ Repo = 'bartowski/Qwen_Qwen3-4B-GGUF'; File = 'Qwen_Qwen3-4B-Q4_K_M.gguf'; ApproxGB = 2.5 }
}
$targets = if ($Size -eq 'all') { '8b','4b' } else { ,$Size }

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
$drive  = (Get-Item $ModelDir).PSDrive
$freeGB = [math]::Round((Get-PSDrive $drive.Name).Free / 1GB, 1)
Write-Host "Models dir: $ModelDir   (free on $($drive.Name): $freeGB GB)" -ForegroundColor Cyan

foreach ($t in $targets) {
    $m    = $catalog[$t]
    $url  = "https://huggingface.co/$($m.Repo)/resolve/main/$($m.File)`?download=true"
    $dest = Join-Path $ModelDir $m.File

    Write-Host "`n=== $t : $($m.File)  (~$($m.ApproxGB) GB) ===" -ForegroundColor Cyan
    # curl.exe ships with Windows 10+. -C - resumes; -L follows the CDN redirect.
    curl.exe -L -C - -o "$dest" "$url"

    if (Test-Path $dest) {
        $sz = [math]::Round((Get-Item $dest).Length / 1GB, 2)
        if ($sz -lt ($m.ApproxGB - 0.5)) {
            Write-Warning "$($m.File) is $sz GB (< expected ~$($m.ApproxGB)) -- may be incomplete; re-run to resume."
        } else {
            Write-Host "Done: $dest  ($sz GB)" -ForegroundColor Green
        }
    } else {
        Write-Warning "No file produced for $t."
    }
}

Write-Host "`nNext: start the 8B brain ->  ./run-mira-small.ps1" -ForegroundColor Green
