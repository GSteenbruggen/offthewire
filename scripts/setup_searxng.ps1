<#
    Start a local SearXNG instance for the agent's web lookup.

    SearXNG runs on YOUR machine. The agent only ever talks to localhost, so
    the loopback guard in ollama_client stays intact; SearXNG is what actually
    reaches out to search engines. No API key, no account, no vendor log.

    Requires Docker Desktop to be running.

        .\scripts\setup_searxng.ps1
        .\scripts\setup_searxng.ps1 -Stop

    This also ships inside the installed application, where it is launched by
    the setup wizard if you tick "set up web lookup". That is why the config
    location is a parameter rather than a path relative to this file: from a
    checkout the config belongs beside the project, but from an install the
    script sits in Program Files next to the exe and "..\searxng-config" would
    put a writable directory somewhere no one would look for it.
#>
param(
    [switch]$Stop,
    [int]$Port = 8080,
    [string]$ConfigDir,
    # Launched from the installer the window closes the instant this returns,
    # taking any error message with it. Hold it open so it can be read.
    [switch]$Pause
)

# NOTE: deliberately NOT "Stop". Docker writes progress and warnings to stderr
# even on success, and in Windows PowerShell 5.1 a native command's stderr under
# ErrorActionPreference=Stop becomes a terminating NativeCommandError -- which
# made this script report "Docker is not running" against a perfectly healthy
# engine. Check $LASTEXITCODE instead of trapping exceptions, and do not
# redirect native stderr.
$ErrorActionPreference = "Continue"
$Name = "searxng-agent"
if (-not $ConfigDir) {
    $ConfigDir = Join-Path $PSScriptRoot "..\searxng-config"
}

# Every exit goes through here so -Pause cannot be forgotten on a new branch.
function Finish([int]$Code) {
    if ($Pause) {
        Write-Host ""
        Read-Host "Press Enter to close"
    }
    exit $Code
}

if ($Stop) {
    Write-Host "Stopping $Name..."
    docker rm -f $Name | Out-Null
    Write-Host "Stopped."
    Finish 0
}

# --- preflight -------------------------------------------------------------

# Checked before it is called, not by inspecting $LASTEXITCODE afterwards: if
# docker.exe does not exist at all, PowerShell raises CommandNotFoundException
# and never sets an exit code, so $LASTEXITCODE still holds whatever the last
# real command left there -- possibly 0, which reads as success.
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Docker is not installed on this machine." -ForegroundColor Red
    Write-Host ""
    Write-Host "Web lookup runs SearXNG in a local Docker container, so it needs"
    Write-Host "Docker Desktop:  https://www.docker.com/products/docker-desktop/"
    Write-Host ""
    Write-Host "Everything else works without it -- only --web is affected."
    Finish 1
}

$serverVersion = docker info --format "{{.ServerVersion}}"
if ($LASTEXITCODE -ne 0 -or -not $serverVersion) {
    Write-Host "Docker is installed but not running." -ForegroundColor Red
    Write-Host "Start Docker Desktop, wait for it to report Running, then re-run this script."
    Finish 1
}
Write-Host "Docker engine $serverVersion"

$existing = docker ps -a --filter "name=$Name" --format "{{.Names}}"
if ($existing) {
    Write-Host "Removing existing $Name container..."
    docker rm -f $Name | Out-Null
}

# --- config ----------------------------------------------------------------
# SearXNG serves JSON only if 'json' is listed under search.formats. It is NOT
# enabled by default, and omitting it is the single most common reason the
# agent's search calls fail with a 403 or an HTML body.

New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null
$SecretKey = -join ((1..64) | ForEach-Object { '{0:x}' -f (Get-Random -Max 16) })

$settings = @"
use_default_settings: true

server:
  secret_key: "$SecretKey"
  limiter: false
  image_proxy: false

search:
  safe_search: 0
  autocomplete: ""
  formats:
    - html
    - json

engines:
  - name: google
    disabled: false
  - name: duckduckgo
    disabled: false
  - name: stackoverflow
    disabled: false
"@

$settingsPath = Join-Path $ConfigDir "settings.yml"
$settings | Out-File -FilePath $settingsPath -Encoding utf8 -NoNewline
Write-Host "Wrote $settingsPath (json format enabled)"

# --- run -------------------------------------------------------------------

$abs = (Resolve-Path $ConfigDir).Path
Write-Host "Starting SearXNG on http://localhost:$Port ..."

# Bind to loopback explicitly. A bare "-p 8080:8080" publishes on 0.0.0.0,
# which makes this an open search relay for anyone on the same network --
# their queries would leave from this machine's address. Ollama binds to
# 127.0.0.1 by default; this needs to be told.
docker run -d `
    --name $Name `
    --restart unless-stopped `
    -p "127.0.0.1:${Port}:8080" `
    -v "${abs}:/etc/searxng:rw" `
    -e "SEARXNG_BASE_URL=http://localhost:$Port/" `
    searxng/searxng:latest | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Host "docker run failed." -ForegroundColor Red
    Finish 1
}

Write-Host "Waiting for it to come up..."
$ok = $false
foreach ($i in 1..30) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:$Port/search?q=test&format=json" `
             -UseBasicParsing -TimeoutSec 5
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
}

if ($ok) {
    Write-Host ""
    Write-Host "SearXNG is up and returning JSON." -ForegroundColor Green
    Write-Host "Run the agent with web lookup:" -ForegroundColor Green
    # The invocation differs by where this copy of the script lives, and
    # printing the wrong one is worse than printing nothing.
    if (Test-Path (Join-Path $PSScriptRoot "OffTheWire.exe")) {
        Write-Host "  OffTheWire --web"
    } else {
        Write-Host "  .venv\Scripts\python.exe src\agent.py <workspace> --web"
    }
    Write-Host ""
    Write-Host "The container restarts with Docker, so this is a one-time setup."
    Finish 0
} else {
    Write-Host ""
    Write-Host "Container started but the JSON endpoint did not respond in 60s." -ForegroundColor Yellow
    Write-Host "Check logs with:  docker logs $Name"
    Finish 1
}
