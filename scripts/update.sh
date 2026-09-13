#!/usr/bin/env bash
# Update OffTheWire to the latest version. POSIX counterpart of update.ps1.
#
#     ./scripts/update.sh             # from a source checkout
#     ./update.sh                     # from an extracted release tarball
#     ./scripts/update.sh --check     # report what would happen, change nothing
#     ./scripts/update.sh --release   # checkout: move to the latest release tag
#                                     # instead of the tip of the current branch
#
# Works out which of the two install shapes it is running in:
#
#   source checkout   .git and .venv next to this script's parent folder.
#                     Fetches, fast-forwards the current branch (or checks out
#                     the newest release tag with --release), reinstalls
#                     requirements.txt into the existing .venv, and re-runs
#                     verify_offline.py so the containment guarantee is
#                     re-proven on the new code.
#
#   release tarball   The OffTheWire binary next to this script (CI puts a
#                     copy of this file in every tarball). Asks GitHub for the
#                     latest release, compares with `OffTheWire --version`,
#                     downloads this platform's tarball, and replaces the
#                     contents of the install folder with it. Saved
#                     conversations live under ~/.local/share (or Library on
#                     macOS), not here, so they are untouched.
#
# THIS SCRIPT GOES ONLINE. It is one of exactly two things in the project that
# do (the other is web lookup, which is off unless asked for), and it talks
# only to github.com. It runs only when you run it; nothing in the agent calls
# it, and there is no background update check.
#
# Downloads are checked against the size the release API reports, which
# catches a truncated transfer but is not a signature.

set -eu

REPO="GSteenbruggen/offthewire"
API_LATEST="https://api.github.com/repos/$REPO/releases/latest"

CHECK=0
RELEASE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --check) CHECK=1 ;;
        --release) RELEASE=1 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

step() { printf '\n=== %s ===\n' "$1"; }
ok()   { printf '  %s\n' "$1"; }
note() { printf '  %s\n' "$1"; }
fail() { printf '  %s\n' "$1" >&2; exit 1; }

HERE="$(cd "$(dirname "$0")" && pwd)"
CHECKOUT_ROOT="$(cd "$HERE/.." 2>/dev/null && pwd || true)"
PACKAGED_BIN="$HERE/OffTheWire"

IS_CHECKOUT=0
if [ -n "$CHECKOUT_ROOT" ] && [ -d "$CHECKOUT_ROOT/.git" ] && [ -f "$CHECKOUT_ROOT/src/_version.py" ]; then
    IS_CHECKOUT=1
fi
IS_PACKAGED=0
if [ -x "$PACKAGED_BIN" ]; then
    IS_PACKAGED=1
fi

if [ "$IS_CHECKOUT" = 0 ] && [ "$IS_PACKAGED" = 0 ]; then
    fail "Cannot tell what to update. Run this from a source checkout's scripts/ folder, or from the folder the release tarball was extracted into."
fi

# --- source checkout -------------------------------------------------------

if [ "$IS_CHECKOUT" = 1 ]; then
    ROOT="$CHECKOUT_ROOT"
    PYTHON="$ROOT/.venv/bin/python"
    [ -x "$PYTHON" ] || fail "No .venv at $ROOT. Create it first (see README > From source)."
    command -v git >/dev/null 2>&1 || fail "git is not on PATH; a checkout cannot be updated without it."

    current_version() {
        sed -n 's/^__version__ *= *"\([^"]*\)".*/\1/p' "$ROOT/src/_version.py"
    }

    step "Source checkout at $ROOT"
    BRANCH="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
    BEFORE="$(git -C "$ROOT" rev-parse --short HEAD)"
    note "version $(current_version), $BRANCH @ $BEFORE"

    # Tracked changes only: untracked files cannot be lost by a fast-forward.
    if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]; then
        echo
        echo "  The checkout has uncommitted changes:"
        git -C "$ROOT" status --porcelain --untracked-files=no | sed 's/^/    /'
        fail "Commit or stash them first; updating over local edits would risk losing them."
    fi

    step "Fetching from origin"
    git -C "$ROOT" fetch --tags --prune origin || fail "git fetch failed (offline, or no access to the remote)."

    if [ "$RELEASE" = 1 ]; then
        # Newest by version number, not by date.
        TARGET="$(git -C "$ROOT" tag --list 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -t. -k1,1V -k2,2n -k3,3n | tail -1)"
        [ -n "$TARGET" ] || fail "No release tags found on the remote."
        TARGET_DESC="release $TARGET"
    else
        if [ "$BRANCH" = "HEAD" ]; then
            fail "The checkout is on a detached HEAD (probably a release tag). Run 'git checkout master' to follow the branch, or pass --release to move to the newest release."
        fi
        TARGET="origin/$BRANCH"
        TARGET_DESC="the tip of $BRANCH"
        git -C "$ROOT" rev-parse --verify --quiet "$TARGET" >/dev/null \
            || fail "Branch '$BRANCH' has no counterpart on origin. Switch to master, or pass --release."
    fi

    TARGET_SHA="$(git -C "$ROOT" rev-parse --short "$TARGET")"
    BEHIND="$(git -C "$ROOT" rev-list --count "HEAD..$TARGET")"
    AHEAD="$(git -C "$ROOT" rev-list --count "$TARGET..HEAD")"

    if [ "$BEHIND" = 0 ]; then
        ok "Already at $TARGET_DESC ($TARGET_SHA). Nothing to update."
        [ "$AHEAD" = 0 ] || note "($AHEAD local commit(s) ahead of it)"
        exit 0
    fi
    note "$BEHIND new commit(s) on $TARGET_DESC ($BEFORE -> $TARGET_SHA)"
    git -C "$ROOT" log --oneline "HEAD..$TARGET" | sed 's/^/    /'

    if [ "$CHECK" = 1 ]; then
        echo
        ok "Run without --check to apply."
        exit 0
    fi

    if [ "$RELEASE" = 0 ] && [ "$AHEAD" != 0 ]; then
        fail "Branch '$BRANCH' has $AHEAD local commit(s) not on origin; a fast-forward is impossible. Rebase or push them, then rerun."
    fi

    step "Updating code"
    if [ "$RELEASE" = 1 ]; then
        git -C "$ROOT" checkout --quiet "$TARGET" || fail "git could not move to $TARGET_DESC."
    else
        git -C "$ROOT" merge --ff-only "$TARGET" || fail "git could not move to $TARGET_DESC."
    fi
    ok "now at $(git -C "$ROOT" rev-parse --short HEAD)"

    step "Updating dependencies"
    "$PYTHON" -m pip install --disable-pip-version-check --quiet -r "$ROOT/requirements.txt" \
        || fail "pip install failed; the code is updated but a dependency is not."
    ok "requirements.txt satisfied"

    step "Re-verifying containment on the new code"
    if ! "$PYTHON" "$ROOT/scripts/verify_offline.py" | tail -3 | sed 's/^/  /'; then
        fail "verify_offline.py FAILED on the updated code. Do not run it until you know why."
    fi

    echo
    ok "Updated to version $(current_version) ($TARGET_DESC)."
    exit 0
fi

# --- release tarball -------------------------------------------------------

step "Packaged build at $HERE"

for tool in curl tar; do
    command -v "$tool" >/dev/null 2>&1 || fail "$tool is required and not on PATH."
done

INSTALLED="$("$PACKAGED_BIN" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
[ -n "$INSTALLED" ] || fail "Could not read the installed version from OffTheWire --version"
note "installed: $INSTALLED"

case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)            SUFFIX="linux-x86_64" ;;
    Darwin-arm64)            SUFFIX="macos-arm64" ;;
    *) fail "No release tarball is published for $(uname -s)/$(uname -m)." ;;
esac

step "Checking the latest release on GitHub"
JSON="$(curl -fsSL -H 'User-Agent: OffTheWire-update' "$API_LATEST")" \
    || fail "Could not reach the GitHub releases API."
LATEST="$(printf '%s' "$JSON" | sed -n 's/.*"tag_name": *"v\{0,1\}\([0-9.]*\)".*/\1/p' | head -1)"
[ -n "$LATEST" ] || fail "Could not read a version from the release API response."
note "latest:    $LATEST"

# Highest of the two by version sort; equal or older means nothing to do.
NEWEST="$(printf '%s\n%s\n' "$INSTALLED" "$LATEST" | sort -t. -k1,1n -k2,2n -k3,3n | tail -1)"
if [ "$NEWEST" = "$INSTALLED" ]; then
    ok "Already up to date."
    exit 0
fi

ASSET="OffTheWire-$LATEST-$SUFFIX.tar.gz"
URL="https://github.com/$REPO/releases/download/v$LATEST/$ASSET"
# The asset's size, from the same API response, for the truncation check.
SIZE="$(printf '%s' "$JSON" | tr -d '\n' | sed -n "s/.*\"name\": *\"$ASSET\"[^}]*\"size\": *\([0-9]*\).*/\1/p" | head -1)"
if [ -z "$SIZE" ]; then
    fail "Release $LATEST has no $ASSET attached (yet). Check https://github.com/$REPO/releases/tag/v$LATEST"
fi
note "tarball:   $ASSET ($((SIZE / 1048576)) MB)"

if [ "$CHECK" = 1 ]; then
    echo
    ok "Update $INSTALLED -> $LATEST is available. Run without --check to install it."
    exit 0
fi

step "Downloading"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
curl -fL -H 'User-Agent: OffTheWire-update' -o "$TMP/$ASSET" "$URL" || fail "Download failed."
GOT="$(wc -c < "$TMP/$ASSET" | tr -d ' ')"
[ "$GOT" = "$SIZE" ] || fail "Download is $GOT bytes but the release lists $SIZE; refusing to unpack a partial file."
ok "saved to $TMP/$ASSET"

step "Installing $LATEST into $HERE"
tar -xzf "$TMP/$ASSET" -C "$TMP" || fail "Could not extract the tarball."
[ -x "$TMP/OffTheWire/OffTheWire" ] || fail "The tarball does not contain OffTheWire/OffTheWire."

# Replace the folder's contents rather than the folder: the user may have
# extracted it somewhere deliberate, and a launcher may point at this path.
# Removing files this script is running from is safe on POSIX -- the shell
# holds the old inode open -- which is why it is delete-then-copy and never
# overwrite-in-place.
find "$HERE" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
cp -R "$TMP/OffTheWire/." "$HERE/"
chmod +x "$HERE/OffTheWire" "$HERE/update.sh" 2>/dev/null || true

echo
ok "Updated: $("$HERE/OffTheWire" --version 2>&1)"
