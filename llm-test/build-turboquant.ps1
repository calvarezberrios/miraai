<#
  build-turboquant.ps1  --  build the TheTom/llama-cpp-turboquant fork with CUDA
  The Linux instructions assumed the fork was pre-cloned & pre-built at
  /root/llama-cpp-turboquant. On Windows we clone to a host folder and compile it
  INSIDE the cuda:12.4.1-devel container (it has the CUDA toolchain). One-time, ~10-20 min.
#>
param(
    [string]$SrcDir = "D:\llama-cpp-turboquant"
)
$repo = "https://github.com/TheTom/llama-cpp-turboquant"

# 1) clone on the host (so the build cache persists between runs)
if (-not (Test-Path (Join-Path $SrcDir ".git"))) {
    Write-Host "Cloning $repo -> $SrcDir ..." -ForegroundColor Cyan
    git clone --depth 1 $repo $SrcDir
    if ($LASTEXITCODE -ne 0) { Write-Warning "git clone failed."; return }
} else {
    Write-Host "Repo already present at $SrcDir (pulling latest)..." -ForegroundColor Cyan
    git -C $SrcDir pull --ff-only
}

# 2) compile inside the CUDA devel container
Write-Host "Building with CUDA inside nvidia/cuda:12.4.1-devel (this takes a while)..." -ForegroundColor Cyan
$buildCmd = @'
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential cmake git libcurl4-openssl-dev ccache >/dev/null
cd /work
# Static build: versioned .so symlinks can't be created on a Windows bind-mount
# (drvfs) -> "Operation not permitted". Static libs (.a) avoid symlinks and give a
# single self-contained llama-server. LLAMA_CURL off: model is always mounted locally.
rm -rf build
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)" --target llama-server
echo "=== build done ==="
ls -la /work/build/bin/llama-server
'@

docker run --rm --gpus all `
  -v ${SrcDir}:/work `
  -w /work `
  nvidia/cuda:12.4.1-devel-ubuntu22.04 `
  bash -lc $buildCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "TurboQuant binary built at $SrcDir\build\bin\llama-server" -ForegroundColor Green
    Write-Host "Next: run-turboquant.ps1" -ForegroundColor Green
} else {
    Write-Warning "Build failed (exit $LASTEXITCODE). Check the compiler output above."
}
