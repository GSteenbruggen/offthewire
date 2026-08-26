"""Where the program reads its code from, and where it is allowed to write.

Those are the same directory when you run from a checkout, and they must not be
when the program is installed. An installed build lives in ``C:\\Program
Files\\OffTheWire``, which a normal user cannot write to -- so a session file
opened next to the executable fails with PermissionError on the first turn, and
the pasted-image directory can never be created at all. The failure arrives at
the worst possible moment, one keystroke into a conversation.

So the two are separated here, in one place, and every writable location in the
program is asked for by name rather than derived from ``__file__``:

    running from source   sessions land in <repo>/sessions, as they always have
    running as an .exe    sessions land in %LOCALAPPDATA%\\OffTheWire

Keeping the source behaviour identical is deliberate: the README, the tests and
years of existing session files all assume ``<repo>/sessions``, and a packaging
change has no business moving them.

``OFFTHEWIRE_HOME`` overrides both, which is what you want for a portable
install on a USB stick, and what the test suite uses to avoid writing into the
real one.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "OffTheWire"
HOME_ENV = "OFFTHEWIRE_HOME"


def is_frozen() -> bool:
    """True when running from a PyInstaller build rather than a checkout.

    PyInstaller sets ``sys.frozen``; ``sys._MEIPASS`` additionally points at the
    unpacked bundle in one-file mode. Checking the former is enough here -- we
    only need to know that ``__file__`` no longer describes a source tree.
    """
    return bool(getattr(sys, "frozen", False))


def install_root() -> Path:
    """The directory the program was launched from.

    Read-only in an installed build, so nothing may be written here. Used for
    locating files that ship *with* the program, never for output.
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """The one directory this program may write to."""
    if override := os.environ.get(HOME_ENV):
        return Path(override).expanduser()
    if not is_frozen():
        # A checkout writes next to itself, exactly as it always did.
        return install_root()
    if base := os.environ.get("LOCALAPPDATA"):
        return Path(base) / APP_NAME
    if sys.platform == "darwin":
        # The platform convention; ~/.local/share exists on a Mac only if
        # something else already broke the convention.
        return Path.home() / "Library" / "Application Support" / APP_NAME
    # Linux, or a stripped environment: the XDG spot.
    xdg = os.environ.get("XDG_DATA_HOME")
    return (Path(xdg) if xdg else Path.home() / ".local" / "share") / APP_NAME


def sessions_dir() -> Path:
    """Saved conversations, pasted images and the input history live here."""
    return data_dir() / "sessions"
