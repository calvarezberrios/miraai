<#
  build-turboquant.ps1  --  build the TheTom/llama-cpp-turboquant fork with CUDA
  The Linux instructions assumed the fork was pre-cloned & pre-built at
  /root/llama-cpp-turboquant. On Windows we clone to a host folder and compile it
  INSIDE the cuda:12.4.1-devel container (it has the CUDA toolchain). One-time, ~10-20 min.
#>
param(
    [string]$SrcDir = "D:\llama-cpp-turboquant",
    # Blackwell (sm_120, e.g. RTX 5050) needs CUDA >=12.8; the desktop's 1660 (sm_75)
    # built fine on 12.4.1. Override with -Image to match your GPU/toolchain.
    [string]$Image  = "nvidia/cuda:12.8.0-devel-ubuntu22.04",
    # Empty = let CMake/ggml pick (desktop default). Set to your GPU's compute capability
    # to force it, e.g. 120 for Blackwell (RTX 50-series) so you don't get "no kernel image".
    [string]$CudaArch = ""
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
Write-Host "Building with CUDA inside $Image (this takes a while)..." -ForegroundColor Cyan
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
cmake -B build -DGGML_CUDA=ON __ARCH__ -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)" --target llama-server
echo "=== build done ==="
ls -la /work/build/bin/llama-server
'@

# Inject optional CUDA arch into the configure line.
$archFlag = if ($CudaArch) { "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch" } else { "" }
$buildCmd = $buildCmd -replace '__ARCH__', $archFlag

# Run the script as a MOUNTED FILE, not as `bash -lc <string>`. On Windows, passing a
# multiline script as a single docker.exe argument is unreliable -- only the first line
# (`set -e`) survives, so the build silently no-ops and still exits 0. Writing the script
# to a temp .sh with LF endings (\r breaks bash: `cd: $'/work\r'`) and running the file
# avoids both traps.
$tmpDir = Join-Path $env:TEMP "tq-build-$PID"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
[System.IO.File]::WriteAllText((Join-Path $tmpDir 'build.sh'), ($buildCmd -replace "`r`n","`n" -replace "`r","`n"))

docker run --rm --gpus all `
  -v ${SrcDir}:/work `
  -v ${tmpDir}:/scratch `
  -w /work `
  $Image `
  bash /scratch/build.sh

Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue

if ($LASTEXITCODE -eq 0) {
    Write-Host "TurboQuant binary built at $SrcDir\build\bin\llama-server" -ForegroundColor Green
    Write-Host "Next: run-turboquant.ps1" -ForegroundColor Green
} else {
    Write-Warning "Build failed (exit $LASTEXITCODE). Check the compiler output above."
}
