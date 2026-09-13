<#
    Update OffTheWire to the latest version.

        .\scripts\update.ps1            # from a source checkout
        update.ps1                      # from the install folder of a packaged build
        update.ps1 -Check               # report what would happen, change nothing
        .\scripts\update.ps1 -Release   # checkout: move to the latest release tag
                                        # instead of the tip of the current branch

    Works out which of the two install shapes it is running in and updates
    that one:

      source checkout   .git and .venv next to this script's parent folder.
                        Fetches, fast-forwards the current branch (or checks
                        out the newest release tag with -Release), reinstalls
                        requirements.txt into the existing .venv, and re-runs
                        verify_offline.py so the containment guarantee is
                        re-proven on the new code before anyone runs it.

      packaged build    OffTheWire.exe next to this script (the installer
                        ships a copy here). Asks GitHub for the latest
                        release, compares it with `OffTheWire.exe --version`,
                        downloads the setup .exe to a temp folder and runs it
                        silently. Inno Setup upgrades in place -- same AppId,
                        same folder, the PATH and Start Menu choices from the
                        original install remembered -- and saved conversations
                        live outside the install folder, so they are untouched.

    THIS SCRIPT GOES ONLINE. It is one of exactly two things in the project
    that do (the other is web lookup, which is off unless asked for), and it
    talks only to github.com: the releases API to learn the newest version and
    the release asset to fetch it, or the git remote for a checkout. It runs
    only when you run it; nothing in the agent calls it, and there is no
    background update check.

    A checkout with uncommitted changes is left alone -- fast-forwarding over
    local edits is how work gets lost -- and the script says what to do.

    Downloads are checked against the size the release API reports, which
    catches a truncated transfer but is not a signature; the installer is
    unsigned, as the README says.
#>
param(
    [switch]$Check,
    [switch]$Release
)

$ErrorActionPreference = "Stop"

$Repo = "GSteenbruggen/offthewire"
$ApiLatest = "https://api.github.com/repos/$Repo/releases/latest"

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Ok($text)   { Write-Host "  $text" -ForegroundColor Green }
function Note($text) { Write-Host "  $text" -ForegroundColor DarkGray }
function Fail($text) { Write-Host "  $text" -ForegroundColor Red; exit 1 }

# Windows PowerShell 5.1 negotiates TLS 1.0 by default, which GitHub refuses.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# --- which shape is this? --------------------------------------------------

$here = $PSScriptRoot
$checkoutRoot = (Resolve-Path (Join-Path $here "..") -ErrorAction SilentlyContinue).Path
$packagedExe = Join-Path $here "OffTheWire.exe"

$isCheckout = $checkoutRoot -and (Test-Path (Join-Path $checkoutRoot ".git")) -and
              (Test-Path (Join-Path $checkoutRoot "src\_version.py"))
$isPackaged = Test-Path $packagedExe

if (-not $isCheckout -and -not $isPackaged) {
    Fail ("Cannot tell what to update. Run this from a source checkout's scripts\ " +
          "folder, or from the folder OffTheWire.exe was installed into.")
}

# --- source checkout -------------------------------------------------------

if ($isCheckout) {
    $root = $checkoutRoot
    $python = Join-Path $root ".venv\Scripts\python.exe"
    if (-not (Test-Path $python)) {
        Fail "No .venv at $root. Create it first (see README > From source)."
    }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Fail "git is not on PATH; a checkout cannot be updated without it."
    }

    function CurrentVersion {
        $text = Get-Content (Join-Path $root "src\_version.py") -Raw
        if ($text -match '__version__\s*=\s*"([^"]+)"') { $Matches[1] } else { "?" }
    }

    Step "Source checkout at $root"
    $branch = (& git -C $root rev-parse --abbrev-ref HEAD).Trim()
    $before = (& git -C $root rev-parse --short HEAD).Trim()
    Note "version $(CurrentVersion), $branch @ $before"

    # Tracked changes only. Untracked files cannot be lost by a fast-forward
    # (git refuses to overwrite them), and a stray notes file should not
    # block an update.
    $dirty = & git -C $root status --porcelain --untracked-files=no
    if ($dirty) {
        Write-Host ""
        Write-Host "  The checkout has uncommitted changes:" -ForegroundColor Yellow
        $dirty | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow }
        Fail "Commit or stash them first; updating over local edits would risk losing them."
    }

    Step "Fetching from origin"
    & git -C $root fetch --tags --prune origin
    if ($LASTEXITCODE -ne 0) { Fail "git fetch failed (offline, or no access to the remote)." }

    if ($Release) {
        # Newest tag by version number, not by date: a re-tagged hotfix on an
        # older line must not win over the actual latest.
        $tags = & git -C $root tag --list "v*" |
                Where-Object { $_ -match '^v\d+\.\d+\.\d+$' } |
                Sort-Object { [version]($_.TrimStart("v")) } -Descending
        if (-not $tags) { Fail "No release tags found on the remote." }
        $target = $tags[0]
        $targetRef = $target
        $targetDesc = "release $target"
    } else {
        if ($branch -eq "HEAD") {
            Fail ("The checkout is on a detached HEAD (probably a release tag). " +
                  "Run 'git checkout master' to follow the branch, or pass -Release " +
                  "to move to the newest release.")
        }
        $target = "origin/$branch"
        $targetRef = $target
        $targetDesc = "the tip of $branch"
        $upstream = & git -C $root rev-parse --verify --quiet $target
        if ($LASTEXITCODE -ne 0) {
            Fail "Branch '$branch' has no counterpart on origin. Switch to master, or pass -Release."
        }
    }

    $targetSha = (& git -C $root rev-parse --short $targetRef).Trim()
    $behind = [int](& git -C $root rev-list --count "HEAD..$targetRef")
    $ahead = [int](& git -C $root rev-list --count "$targetRef..HEAD")

    if ($behind -eq 0) {
        Ok "Already at $targetDesc ($targetSha). Nothing to update."
        if ($ahead -gt 0) { Note "($ahead local commit(s) ahead of it)" }
        exit 0
    }
    Note "$behind new commit(s) on $targetDesc ($before -> $targetSha)"
    & git -C $root log --oneline "HEAD..$targetRef" | ForEach-Object { Note "  $_" }

    if ($Check) {
        Write-Host ""
        Ok "Run without -Check to apply."
        exit 0
    }

    if (-not $Release -and $ahead -gt 0) {
        Fail ("Branch '$branch' has $ahead local commit(s) not on origin; a fast-forward is " +
              "impossible. Rebase or push them, then rerun.")
    }

    Step "Updating code"
    if ($Release) {
        & git -C $root checkout --quiet $target
    } else {
        & git -C $root merge --ff-only $targetRef
    }
    if ($LASTEXITCODE -ne 0) { Fail "git could not move to $targetDesc." }
    Ok "now at $((& git -C $root rev-parse --short HEAD).Trim())"

    Step "Updating dependencies"
    & $python -m pip install --disable-pip-version-check --quiet -r (Join-Path $root "requirements.txt")
    if ($LASTEXITCODE -ne 0) { Fail "pip install failed; the code is updated but a dependency is not." }
    Ok "requirements.txt satisfied"

    Step "Re-verifying containment on the new code"
    & $python (Join-Path $root "scripts\verify_offline.py") | Select-Object -Last 3 | ForEach-Object { Note $_ }
    if ($LASTEXITCODE -ne 0) {
        Fail "verify_offline.py FAILED on the updated code. Do not run it until you know why."
    }

    Write-Host ""
    Ok "Updated to version $(CurrentVersion) ($targetDesc)."
    exit 0
}

# --- packaged build --------------------------------------------------------

Step "Packaged build at $here"

# No stderr redirect on native commands: under ErrorActionPreference=Stop,
# Windows PowerShell 5.1 turns any stderr line into a terminating error.
$verOut = (& $packagedExe --version) -join " "
if ($LASTEXITCODE -ne 0 -or $verOut -notmatch '(\d+\.\d+\.\d+)') {
    Fail "Could not read the installed version from OffTheWire.exe --version"
}
$installed = $Matches[1]
Note "installed: $installed"

Step "Checking the latest release on GitHub"
# Not "$release": variable names are case-insensitive and that is the
# [switch] parameter above, which cannot hold an API response.
try {
    $latestRelease = Invoke-RestMethod -Uri $ApiLatest -Headers @{ "User-Agent" = "OffTheWire-update" } -UseBasicParsing
} catch {
    Fail "Could not reach the GitHub releases API: $($_.Exception.Message)"
}
$latest = ($latestRelease.tag_name -replace '^v', '')
if ($latest -notmatch '^\d+\.\d+\.\d+$') { Fail "Unexpected release tag: $($latestRelease.tag_name)" }
Note "latest:    $latest"

if ([version]$latest -le [version]$installed) {
    Ok "Already up to date."
    exit 0
}

$asset = $latestRelease.assets | Where-Object { $_.name -eq "OffTheWire-Setup-$latest.exe" } | Select-Object -First 1
if (-not $asset) {
    Fail ("Release $latest has no Windows installer attached (yet). " +
          "Check https://github.com/$Repo/releases/tag/$($latestRelease.tag_name)")
}
Note "installer: $($asset.name) ($([math]::Round($asset.size / 1MB, 1)) MB)"

if ($Check) {
    Write-Host ""
    Ok "Update $installed -> $latest is available. Run without -Check to install it."
    exit 0
}

Step "Downloading"
$tmp = Join-Path $env:TEMP "OffTheWire-update"
New-Item -ItemType Directory -Force $tmp | Out-Null
$setup = Join-Path $tmp $asset.name
try {
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $setup -UseBasicParsing `
        -Headers @{ "User-Agent" = "OffTheWire-update" }
} catch {
    Fail "Download failed: $($_.Exception.Message)"
}
$got = (Get-Item $setup).Length
if ($got -ne [int64]$asset.size) {
    Remove-Item $setup -Force
    Fail "Download is $got bytes but the release lists $($asset.size); refusing to run a partial file."
}
Ok "saved to $setup"

Step "Installing $latest"
Note "The installer upgrades in place; your PATH and Start Menu choices are kept."
Note "Close any running OffTheWire session first if the installer asks."
# /SILENT shows a progress bar but no pages; /SP- skips the 'This will install'
# prompt; /CLOSEAPPLICATIONS asks running instances to close rather than
# failing on a locked exe; /NORESTART because nothing here needs a reboot.
$proc = Start-Process -FilePath $setup -Wait -PassThru `
        -ArgumentList "/SILENT", "/SP-", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS"
if ($proc.ExitCode -ne 0) {
    Fail "The installer exited with code $($proc.ExitCode). The download is still at $setup."
}
Remove-Item $setup -Force -ErrorAction SilentlyContinue

$after = (& $packagedExe --version) -join " "
Write-Host ""
Ok "Updated: $after"
Note "Open a new terminal if the old version still answers to 'OffTheWire'."
