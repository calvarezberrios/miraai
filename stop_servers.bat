@echo off
REM ============================================================================
REM  stop_servers.bat  --  fully stop Mira + the GPT-SoVITS TTS server.
REM
REM  Kills, with their child processes (/T):
REM    - GPT-SoVITS  (python running api_v2.py, :9880)
REM    - Mira        (python running main.py, + any Kokoro/RVC subprocesses)
REM  Their console windows close on their own once the process dies.
REM
REM  Deliberately targets ONLY python processes whose command line is api_v2.py or
REM  main.py, so it never touches Ollama (the local embeddings server) or anything
REM  else. The laptop's LLM server is unaffected (different machine).
REM ============================================================================
echo Stopping Mira (main.py) and GPT-SoVITS (api_v2.py)...

powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('python.exe','pythonw.exe') -and $_.CommandLine -and ($_.CommandLine -match 'api_v2\.py' -or $_.CommandLine -match 'main\.py') } | ForEach-Object { Write-Host ('  killing PID {0}  ({1})' -f $_.ProcessId, $_.Name); taskkill /PID $_.ProcessId /T /F | Out-Null }"

REM Safety net: if anything still holds the SoVITS port, free it.
for /f "tokens=5" %%P in ('netstat -ano ^| findstr :9880 ^| findstr LISTENING') do taskkill /PID %%P /T /F >nul 2>&1

echo Done. (Ollama embeddings and the laptop LLM are left running.)
