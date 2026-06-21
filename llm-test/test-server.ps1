<#
  test-server.ps1  --  quick smoke test once a server is running on :8080
  Hits the OpenAI-compatible endpoint and prints the reply.
#>
param([int]$Port = 8080, [string]$Prompt = "In one sentence, what is TurboQuant KV cache?")

$body = @{
    model    = "local"
    messages = @(@{ role = "user"; content = $Prompt })
    stream   = $false
} | ConvertTo-Json -Depth 5

try {
    $r = Invoke-RestMethod -Uri "http://localhost:$Port/v1/chat/completions" `
            -Method Post -ContentType "application/json" -Body $body -TimeoutSec 600
    Write-Host "Reply:" -ForegroundColor Green
    Write-Host $r.choices[0].message.content
} catch {
    Write-Warning "Request failed: $($_.Exception.Message)"
    Write-Host "Is the server up? Check  http://localhost:$Port/health" -ForegroundColor Yellow
}
