<#
  run-stock.ps1  --  llama.cpp server WITHOUT TurboQuant
  Uses the prebuilt CUDA image; nothing to compile. Good first smoke test.

  Original Linux flags preserved; Windows additions:
    -v D:\models:/models        (Docker Desktop translates the Windows path)
    -p 8080:8080 + --host 0.0.0.0  (so you can reach it at http://localhost:8080)
    --ulimit memlock=-1         (lets --mlock pin the 20.75 GB model; LXC line N/A on Windows)
#>
param(
    [string]$ModelDir = "D:\models",
    [string]$Model    = "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
    [int]   $CtxSize  = 128000,
    [int]   $Port     = 8080
)

if (-not (Test-Path (Join-Path $ModelDir $Model))) {
    Write-Warning "Model not found: $(Join-Path $ModelDir $Model) -- run 02-download-model.ps1 first."
    return
}

Write-Host "Starting llama.cpp (stock CUDA, ctx=$CtxSize) on http://localhost:$Port ..." -ForegroundColor Cyan
docker run --rm -it `
  --gpus all `
  --cap-add=IPC_LOCK `
  --ulimit memlock=-1 `
  -p ${Port}:8080 `
  -v ${ModelDir}:/models `
  ghcr.io/ggml-org/llama.cpp:server-cuda `
    -m /models/$Model `
    --host 0.0.0.0 --port 8080 `
    -ngl 999 `
    --n-cpu-moe 36 `
    --no-mmap `
    --mlock `
    --ctx-size $CtxSize
