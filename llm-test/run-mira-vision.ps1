<#
  run-mira-vision.ps1  --  LAPTOP BRAIN SERVER with EYES (Qwen2.5-VL).

  Serves a Qwen2.5-VL model on llama.cpp so the laptop GPU is BOTH Mira's chat brain
  AND her vision model. One model does both because the 8 GB laptop GPU can't hold a
  separate 8B text model + a 7B VL model at once — so the VL replaces the text turbo on
  the same :8080 endpoint Mira already points at.

    -Size 7b  (DEFAULT)  Qwen2.5-VL-7B-Instruct Q4_K_M (~4.7 GB) + mmproj (~1.3 GB)
    -Size 3b             Qwen2.5-VL-3B-Instruct Q4_K_M (~2.2 GB) + mmproj (~0.9 GB)  -- safer on 8 GB

  Vision works because llama-server, given --mmproj, accepts images in chat completions
  (the OpenAI `image_url` content the desktop's stream_vision.py sends).

  DESKTOP SIDE (Mira): vision auto-uses the chat endpoint, so just run with --vision:
      python main.py --discord --twitch --host --vision --game-audio
    and confirm in .env (or env):
      OLLAMA_BASE_URL=http://<this-laptop-ip>:8080/v1
      MIRA_MODEL=qwen2.5-vl-7b           # chat model name = what this server serves
      MIRA_VISION_MODEL=qwen2.5-vl-7b    # same model for captions
      # MIRA_VISION_BASE_URL is optional; defaults to OLLAMA_BASE_URL.

  PREREQS:
    1. turboquant binary built once (./build-turboquant.ps1).
    2. Model + mmproj GGUFs in $ModelDir (see ./download-vision.ps1 or the Hugging Face
       links in VISION_SETUP.md). BOTH files are required.
    3. Firewall rule for :8080 (see run-turbo-server.ps1).
#>
param(
    [ValidateSet('7b','3b')][string]$Size = '7b',
    [string]$ModelDir = "D:\models",
    [string]$LlamaDir = "D:\llama-cpp-turboquant",
    [string]$Image    = "nvidia/cuda:12.4.1-devel-ubuntu22.04",
    [int]   $CtxSize  = 16384,   # VL images eat context; 16k leaves room for the KV + a frame.
    [int]   $Port     = 8080
)

# Model + its matching vision projector (mmproj). Filenames follow the ggml-org / common
# GGUF repos; override by editing here if your download named them differently.
$models = @{
    '7b' = @{ model = 'Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf'; mmproj = 'mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf' }
    '3b' = @{ model = 'Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf'; mmproj = 'mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf' }
}
$Model  = $models[$Size].model
$MMProj = $models[$Size].mmproj

# Laptop has no D:\ — fall back to C:\ paths and the Blackwell CUDA 12.8 image (sm_120).
if (-not (Test-Path 'D:\')) {
    if (-not $PSBoundParameters.ContainsKey('ModelDir')) { $ModelDir = 'C:\models' }
    if (-not $PSBoundParameters.ContainsKey('LlamaDir')) { $LlamaDir = 'C:\llama-cpp-turboquant' }
    if (-not $PSBoundParameters.ContainsKey('Image'))    { $Image    = 'nvidia/cuda:12.8.0-devel-ubuntu22.04' }
    Write-Host "No D:\ drive -- using laptop paths ($ModelDir) and the 12.8 image." -ForegroundColor DarkCyan
}

foreach ($f in @($Model, $MMProj)) {
    if (-not (Test-Path (Join-Path $ModelDir $f))) {
        Write-Warning "Missing GGUF: $(Join-Path $ModelDir $f)"
        Write-Host "Download the model AND its mmproj first -- see ./download-vision.ps1 or VISION_SETUP.md" -ForegroundColor Yellow
        return
    }
}

docker rm -f llama-turbo 2>$null | Out-Null
# Same as run-mira-small, plus --mmproj for the vision encoder. -ngl 99 = full GPU offload.
docker run -d --name llama-turbo `
  --gpus all --ulimit memlock=-1 --cap-add=IPC_LOCK `
  -p ${Port}:8080 `
  -v ${LlamaDir}:/llama `
  -v ${ModelDir}:/models `
  $Image `
  /llama/build/bin/llama-server `
    -m /models/$Model `
    --mmproj /models/$MMProj `
    --host 0.0.0.0 --port 8080 `
    --cache-type-k turbo4 --cache-type-v turbo3 `
    -ngl 99 --jinja `
    -c $CtxSize | Out-Null

Write-Host "Loading $Model (+ vision $MMProj) @ ${CtxSize} ctx on GPU (~30-60s)..." -ForegroundColor Cyan
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 5
    if (-not (docker ps --filter "name=llama-turbo" --format "{{.Status}}")) {
        Write-Warning "Container exited early -- check: docker logs llama-turbo"
        Write-Host "If it OOMed on the 8 GB GPU, try the 3B:  ./run-mira-vision.ps1 -Size 3b" -ForegroundColor Yellow
        return
    }
    try {
        $r = Invoke-WebRequest -UseBasicParsing "http://localhost:$Port/health" -TimeoutSec 3
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
}
if ($ready) {
    Write-Host "Ready. Brain+vision serving at http://localhost:$Port/v1  (model: $Model)" -ForegroundColor Green
    Write-Host "On the desktop set MIRA_MODEL / MIRA_VISION_MODEL to this model and run with --vision." -ForegroundColor Green
} else {
    Write-Warning "Server didn't report healthy in time -- check: docker logs llama-turbo"
}
