<#
    Build the distributable: OffTheWire.exe, then the setup .exe around it.

        .\scripts\build_installer.ps1
        .\scripts\build_installer.ps1 -SkipApp      # installer only, reuse the app build
        .\scripts\build_installer.ps1 -Clean        # discard previous build output first

    Two stages, because they fail for different reasons and you want to know
    which one did:

        1. PyInstaller  src\agent.py      -> build\dist\OffTheWire\
        2. Inno Setup   installer\*.iss   -> build\OffTheWire-Setup-X.Y.Z.exe

    Neither tool is a runtime dependency. The offline guarantee is unaffected:
    this script needs the network only if a build tool is missing, and the
    built program still talks to nothing but localhost.
#>
param(
    [switch]$SkipApp,
    [switch]$Clean
)

# NOTE: deliberately NOT "Stop", for the same reason setup_searxng.ps1 says so.
# PyInstaller writes its entire progress log to stderr even on a clean build,
# and in Windows PowerShell 5.1 a native command's stderr under
# ErrorActionPreference=Stop is turned into a terminating NativeCommandError --
# so a successful build reports itself as a failure on its first INFO line.
# Check $LASTEXITCODE instead, and never redirect native stderr.
$ErrorActionPreference = "Continue"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$spec = Join-Path $root "installer\OffTheWire.spec"
$iss = Join-Path $root "installer\OffTheWire.iss"
$dist = Join-Path $root "build\dist\OffTheWire"
$work = Join-Path $root "build\work"

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Fail($text) { Write-Host $text -ForegroundColor Red; exit 1 }

# --- preflight -------------------------------------------------------------

if (-not (Test-Path $python)) { Fail "No venv at $python. Create it and install requirements.txt first." }

if (-not $SkipApp) {
    # A missing build tool should say so here, not 200 lines into a traceback.
    & $python -c "import PyInstaller" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Fail "PyInstaller is not installed. Run:`n  $python -m pip install pyinstaller"
    }
}

# winget installs Inno Setup per-user or per-machine depending on scope, so
# look in both rather than assuming Program Files.
$isccCandidates = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
)
$iscc = $isccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) {
    Fail ("Inno Setup not found. Install it with:`n" +
          "  winget install --id JRSoftware.InnoSetup`n" +
          "Looked in:`n  " + ($isccCandidates -join "`n  "))
}

if ($Clean -and (Test-Path (Join-Path $root "build"))) {
    Step "Cleaning previous build output"
    Remove-Item (Join-Path $root "build") -Recurse -Force
}

# --- 1. the application ----------------------------------------------------

if ($SkipApp) {
    if (-not (Test-Path (Join-Path $dist "OffTheWire.exe"))) {
        Fail "-SkipApp was passed but there is no build at $dist."
    }
    Step "Reusing existing app build"
} else {
    Step "Building OffTheWire.exe (PyInstaller)"
    & $python -m PyInstaller --noconfirm --distpath (Join-Path $root "build\dist") --workpath $work $spec
    if ($LASTEXITCODE -ne 0) { Fail "PyInstaller failed." }
}

$exe = Join-Path $dist "OffTheWire.exe"
if (-not (Test-Path $exe)) { Fail "Expected $exe to exist after the build." }

# Smoke test before packaging: an app that cannot start is not worth wrapping,
# and --help exercises the whole import graph without needing Ollama.
Step "Smoke testing the built exe"
$out = & $exe --help 2>&1
if ($LASTEXITCODE -ne 0) { Fail "The built exe failed to run --help:`n$out" }
Write-Host "  starts and parses arguments" -ForegroundColor Green

$appSize = [math]::Round((Get-ChildItem $dist -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
Write-Host "  application: $appSize MB" -ForegroundColor Green

# Windows still refuses paths over 260 characters unless long-path support is
# switched on machine-wide. PyInstaller bundles some genuinely deep files --
# lxml's isoschematron\resources\xsl\iso-schematron-xslt1\ is the worst -- so a
# project checked out somewhere deep blows the limit while packaging. Inno's
# only comment on that is "The system cannot find the path specified", which
# sends you looking for a missing file that is right there. Say it plainly.
$longest = Get-ChildItem $dist -Recurse -File |
           Sort-Object { $_.FullName.Length } -Descending |
           Select-Object -First 1
if ($longest -and $longest.FullName.Length -ge 260) {
    Fail ("A bundled file exceeds Windows' 260-character path limit:`n" +
          "  $($longest.FullName.Length) chars  $($longest.FullName)`n`n" +
          "The project is checked out too deep for the installer to package.`n" +
          "Move it somewhere shorter (e.g. C:\src\Ollama-MCP) and rebuild.")
}
if ($longest -and $longest.FullName.Length -ge 240) {
    Write-Host ("  WARNING: longest bundled path is $($longest.FullName.Length) chars; " +
                "the limit is 260. Consider a shorter checkout path.") -ForegroundColor Yellow
}

# --- 2. the installer ------------------------------------------------------

Step "Building the installer (Inno Setup)"
# Absolute source path rather than the script's relative default, so the files
# are not reached through "installer\..\" -- see the MAX_PATH note above.
& $iscc /Q "/DSourceDir=$((Resolve-Path $dist).Path)" $iss
if ($LASTEXITCODE -ne 0) { Fail "Inno Setup failed." }

$setup = Get-ChildItem (Join-Path $root "build") -Filter "OffTheWire-Setup-*.exe" |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $setup) { Fail "Inno Setup reported success but produced no installer." }

$setupSize = [math]::Round($setup.Length / 1MB, 1)
Write-Host "`nBuilt $($setup.Name)  ($setupSize MB)" -ForegroundColor Green
Write-Host "  $($setup.FullName)"
Write-Host ""
Write-Host "Installs per-user into %LOCALAPPDATA%\Programs\OffTheWire - no admin needed."
Write-Host "Conversations are NOT written to disk unless the agent is run with --save;"
Write-Host "when it is, they go to %LOCALAPPDATA%\OffTheWire and survive uninstall unless you say otherwise."
