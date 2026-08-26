"""Images as input: getting one out of the clipboard or off disk and onto the wire.

A terminal has no textbox to drop a picture into, so "attach an image" has to be
assembled from the three things a terminal *can* observe:

  * **Clipboard.** Alt+V asks Windows for the image on the clipboard and writes
    it next to the session as a PNG. This is the screenshot path -- Win+Shift+S
    then Alt+V.
  * **Drag and drop.** Dropping a file on Windows Terminal types its path into
    the line, quoted if it contains spaces. Nothing special is needed beyond
    recognising that a path to a real image is an attachment, not prose.
  * **A path you typed**, or an ``[image: ...]`` marker left by the paste key.

All three end up as a file path, which is the only representation the rest of
the agent has to know about.

Two decisions worth stating:

``ollama`` wants base64 in the message's ``images`` array, so a 2MB screenshot
becomes ~2.7MB of text on every turn it stays in context. The bytes therefore
live in memory only; sessions on disk record the *path*, and a resumed session
re-reads it. A session file that grew by 3MB per screenshot would be unusable.

Format is decided by the file's magic bytes, never its extension. Ollama rejects
what it cannot decode with an opaque error, and a .png that is really a .webp is
common enough (browsers do it) that trusting the name would produce exactly that
error for a file the user can plainly see is an image.

Nothing here talks to the network.
"""

from __future__ import annotations

import base64
import re
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

# What Ollama's vision preprocessors accept. BMP and TIFF are deliberately
# absent: they decode in some builds and not others, and a silent "the model
# ignored my image" is worse than being told to convert it.
SUPPORTED = {"png", "jpeg", "gif", "webp"}

# Extensions we will treat as an attachment when we see one in a line of text.
# Wider than SUPPORTED on purpose -- a dropped .bmp should get a clear message
# about the format, not be silently read as prose.
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff",
}

MAX_IMAGE_BYTES = 20 * 1024 * 1024
LARGE_IMAGE_BYTES = 4 * 1024 * 1024

# Vision models turn an image into a fixed-ish block of tokens rather than
# anything proportional to the base64 length. This is only used to keep the
# context estimate honest between requests; the real count arrives with the
# next response and replaces it.
TOKENS_PER_IMAGE = 800


class ImageError(ValueError):
    """A file that cannot be sent, with a message meant for the user."""


# ------------------------------------------------------------------ inspection


def sniff(data: bytes) -> str:
    """Identify an image by its leading bytes. Returns "" if it is not one."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:2] == b"BM":
        return "bmp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    return ""


def _jpeg_size(data: bytes) -> tuple[int, int]:
    """Walk the marker segments to the frame header.

    The dimensions are not at a fixed offset in a JPEG -- EXIF thumbnails and
    colour profiles sit in front of them and vary in length -- so the segment
    chain has to be followed.
    """
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        # standalone markers: no length field to skip
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg = int.from_bytes(data[i + 2 : i + 4], "big")
        # SOF0..SOF15, excluding the huffman/arithmetic tables interleaved in
        # the same numeric range
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height = int.from_bytes(data[i + 5 : i + 7], "big")
            width = int.from_bytes(data[i + 7 : i + 9], "big")
            return width, height
        if seg < 2:
            break
        i += 2 + seg
    return 0, 0


def dimensions(data: bytes, kind: str) -> tuple[int, int]:
    """Width and height, or (0, 0) when they cannot be read cheaply.

    Only ever used for the confirmation line printed to the user, so an unknown
    size degrades to not showing one rather than to an error.
    """
    try:
        if kind == "png":
            return struct.unpack(">II", data[16:24])
        if kind == "jpeg":
            return _jpeg_size(data)
        if kind == "gif":
            return struct.unpack("<HH", data[6:10])
        if kind == "bmp":
            w, h = struct.unpack("<ii", data[18:26])
            return w, abs(h)
        if kind == "webp":
            chunk = data[12:16]
            if chunk == b"VP8X":
                return (
                    1 + int.from_bytes(data[24:27], "little"),
                    1 + int.from_bytes(data[27:30], "little"),
                )
            if chunk == b"VP8 ":
                return (
                    int.from_bytes(data[26:28], "little") & 0x3FFF,
                    int.from_bytes(data[28:30], "little") & 0x3FFF,
                )
    except Exception:
        pass
    return 0, 0


def human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024.0:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


# ----------------------------------------------------------------- attachments


@dataclass
class Attachment:
    """One image, loaded and ready to go on the wire."""

    path: Path
    data: bytes
    kind: str
    width: int = 0
    height: int = 0

    @property
    def b64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    @property
    def summary(self) -> str:
        size = human_bytes(len(self.data))
        if self.width and self.height:
            return f"{self.kind} · {self.width}×{self.height} · {size}"
        return f"{self.kind} · {size}"

    def describe(self) -> str:
        return f"{self.path.name}  ({self.summary})"


def load(path: Path | str) -> Attachment:
    """Read one image, or raise ImageError saying why it cannot be sent."""
    p = Path(path).expanduser()
    try:
        if not p.is_file():
            raise ImageError(f"{p} is not a file")
        size = p.stat().st_size
        if size == 0:
            raise ImageError(f"{p.name} is empty")
        if size > MAX_IMAGE_BYTES:
            raise ImageError(
                f"{p.name} is {human_bytes(size)}; the limit is "
                f"{human_bytes(MAX_IMAGE_BYTES)}. Scale it down first."
            )
        data = p.read_bytes()
    except ImageError:
        raise
    except OSError as e:
        raise ImageError(f"could not read {p.name}: {e}") from e

    kind = sniff(data)
    if not kind:
        raise ImageError(f"{p.name} is not an image file (its contents are not one)")
    if kind not in SUPPORTED:
        raise ImageError(
            f"{p.name} is {kind.upper()}, which Ollama's vision models do not "
            f"reliably decode. Save it as PNG or JPEG."
        )

    w, h = dimensions(data, kind)
    return Attachment(path=p, data=data, kind=kind, width=w, height=h)


def load_all(paths: list[Path]) -> tuple[list[Attachment], list[str]]:
    """Load what can be loaded; return the rest as messages, not exceptions.

    One unreadable file should not discard the four alongside it, and it should
    certainly not discard the sentence the user typed with them.
    """
    ok: list[Attachment] = []
    errors: list[str] = []
    for p in paths:
        try:
            ok.append(load(p))
        except ImageError as e:
            errors.append(str(e))
    return ok, errors


# ------------------------------------------------------- references in a line


MARKER = re.compile(r"\[(?:image|img)\s*:\s*([^\]]+?)\s*\]", re.IGNORECASE)
QUOTED = re.compile(r"""(["'])(.+?)\1""")
# An unquoted run ending in an image extension. Stops at whitespace, so a path
# with spaces only works quoted -- which is exactly how a drag-and-drop and a
# shell-completed path both arrive.
BARE = re.compile(
    r"(?<![\w\"'])((?:[A-Za-z]:[\\/]|\.{0,2}[\\/]|~[\\/])?[^\s\"'<>|]+"
    r"\.(?:png|jpe?g|gif|webp|bmp|tiff?))(?![\w])",
    re.IGNORECASE,
)


def marker(path: Path | str) -> str:
    """The text form the paste key inserts into the line."""
    return f"[image: {path}]"


def _normalize(raw: str) -> str:
    """Strip what a sentence puts around a path but a filesystem does not.

    Trailing punctuation matters more than it looks: "see shot.png." leaves a
    name whose suffix is "." rather than ".png", which would fail the extension
    check for a file that exists and is perfectly valid.
    """
    return raw.strip().strip("\"'").rstrip(".,;:!?")


def _resolve(raw: str, base: Path | None) -> Path | None:
    """Turn a fragment of a typed line into a real image path, or None."""
    raw = _normalize(raw)
    if not raw:
        return None
    if raw.lower().startswith("file:///"):
        from urllib.parse import unquote, urlparse

        raw = unquote(urlparse(raw).path).lstrip("/")
    p = Path(raw).expanduser()
    if not p.is_absolute() and base is not None:
        p = base / p
    try:
        return p if p.is_file() else None
    except OSError:
        # An over-long or malformed path is just not a file; it is not an error
        # worth surfacing when the text was probably prose all along.
        return None


@dataclass
class Extraction:
    """What a line of input turned out to contain.

    The spans are kept rather than a finished string because how a reference
    should read depends on something decided later: whether the file loaded,
    and whether the model can see images at all. Rendering eagerly produced
    "[image 3: c.png]" for a turn that ended up sending two.
    """

    text: str
    refs: list[Path]
    spans: list[tuple[int, int, Path]]  # (start, end, path) into `text`

    @property
    def has_images(self) -> bool:
        return bool(self.refs)

    def render(self, keep: list[Path] | None = None) -> str:
        """Rewrite the line for the model.

        References in ``keep`` become "[image N: name.png]", numbered to match
        the order the images are attached in. Everything else falls back to its
        path, which is the only useful thing to say about a picture that is not
        being sent.
        """
        keep = self.refs if keep is None else keep
        index = {str(p): i + 1 for i, p in enumerate(keep)}
        out, cursor = [], 0
        for start, end, path in self.spans:
            out.append(self.text[cursor:start])
            n = index.get(str(path))
            out.append(f"[image {n}: {path.name}]" if n else str(path))
            cursor = end
        out.append(self.text[cursor:])
        return "".join(out).strip()


def extract(text: str, base: Path | None = None) -> Extraction:
    """Pull image references out of a user turn.

    A marker is always an attachment. A bare or quoted path is one only if it
    actually exists on disk and looks like an image -- "the bug is in logo.png"
    stays prose when there is no such file, which is the safe way round.
    """
    refs: list[Path] = []
    spans: list[tuple[int, int, Path]] = []
    seen: dict[str, Path] = {}

    def take(raw: str) -> Path | None:
        if Path(_normalize(raw)).suffix.lower() not in IMAGE_EXTENSIONS:
            return None
        p = _resolve(raw, base)
        if p is None:
            return None
        try:
            key = str(p.resolve()).lower()
        except OSError:
            key = str(p).lower()
        if key in seen:
            # The same file named twice in one turn shares one label and is
            # sent once; two copies of a screenshot is pure context.
            return seen[key]
        seen[key] = p
        refs.append(p)
        return p

    def scan(pattern: re.Pattern, group: int) -> None:
        for m in pattern.finditer(text):
            start, end = m.span()
            if any(start < e and s < end for s, e, _ in spans):
                continue  # already claimed by an earlier, more explicit pattern
            if (p := take(m.group(group))) is not None:
                spans.append((start, end, p))

    # Explicit markers first, so a path inside one is not also matched bare.
    scan(MARKER, 1)
    scan(QUOTED, 2)
    scan(BARE, 1)
    spans.sort()

    # Numbering follows the order references appear in the line, not the order
    # the patterns happened to match in.
    refs.sort(key=lambda p: next(s for s, _, q in spans if q == p))

    return Extraction(text=text, refs=refs, spans=spans)


# -------------------------------------------------------------- PNG encoding


def _png_chunk(tag: bytes, body: bytes) -> bytes:
    return (
        struct.pack(">I", len(body))
        + tag
        + body
        + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
    )


def encode_png(width: int, height: int, rows: list[bytes], channels: int) -> bytes:
    """Minimal PNG writer for 8-bit RGB/RGBA.

    Pillow would do this in a line, but it is not a dependency of this project
    and adding one to paste a screenshot is a poor trade. Filter type 0 (none)
    on every row costs some compression ratio and no correctness.
    """
    raw = b"".join(b"\x00" + r for r in rows)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6 if channels == 4 else 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw, 6))
        + _png_chunk(b"IEND", b"")
    )


def dib_to_png(dib: bytes) -> bytes:
    """Convert a Windows device-independent bitmap to PNG.

    CF_DIB is what the Snipping Tool, Paint and most Win32 apps put on the
    clipboard, and it is a raw pixel buffer with a header -- no file signature,
    bottom-up row order, BGR channel order, rows padded to four bytes.

    Only the 24- and 32-bit uncompressed forms are handled, which is everything
    a screenshot has ever produced here. Anything else raises rather than
    guessing at a palette and writing a wrong-coloured image.
    """
    if len(dib) < 40:
        raise ImageError("the clipboard bitmap is truncated")

    header_size = struct.unpack_from("<I", dib, 0)[0]
    width, height = struct.unpack_from("<ii", dib, 4)
    bit_count = struct.unpack_from("<H", dib, 14)[0]
    compression = struct.unpack_from("<I", dib, 16)[0]
    clr_used = struct.unpack_from("<I", dib, 32)[0]

    if bit_count not in (24, 32):
        raise ImageError(
            f"the clipboard holds a {bit_count}-bit bitmap, which cannot be "
            f"converted here. Save it to a PNG and attach the file instead."
        )
    # 0 = BI_RGB, 3 = BI_BITFIELDS. Screenshots use one or the other; anything
    # else is RLE or an embedded JPEG/PNG payload we should not be parsing.
    if compression not in (0, 3):
        raise ImageError("the clipboard bitmap is in a compressed form we cannot read")

    top_down = height < 0
    height = abs(height)
    if width <= 0 or height <= 0:
        raise ImageError("the clipboard bitmap has no size")

    palette = clr_used * 4 if bit_count <= 8 else 0
    # BITMAPINFOHEADER puts the BI_BITFIELDS masks *after* the header; the V4/V5
    # headers carry them inside it, so their pixel data starts right after.
    masks = 12 if (compression == 3 and header_size == 40) else 0
    offset = header_size + palette + masks

    stride = ((width * bit_count + 31) // 32) * 4
    needed = offset + stride * height
    if len(dib) < needed:
        raise ImageError("the clipboard bitmap is truncated")

    step = bit_count // 8
    has_alpha = bit_count == 32
    rows: list[bytearray] = []
    any_alpha = False

    for y in range(height):
        # DIBs are stored bottom-up unless the height is negative.
        src = y if top_down else height - 1 - y
        start = offset + src * stride
        line = dib[start : start + width * step]
        # Strided slice assignment rather than a per-pixel loop: a 4K screenshot
        # is 8M pixels, and iterating those in Python turns a keypress into a
        # multi-second pause. Swapping two channel slices does the same work
        # inside the interpreter's C loops. Green (and alpha) are already in
        # the right column, so only B and R move.
        out = bytearray(line)
        out[0::step] = line[2::step]
        out[2::step] = line[0::step]
        if has_alpha and not any_alpha and max(line[3::4]):
            any_alpha = True
        rows.append(out)

    if has_alpha and not any_alpha:
        # A 32-bit DIB whose alpha channel is entirely zero is padding, not a
        # fully transparent image -- GDI leaves the fourth byte untouched. Taken
        # literally it produces a PNG that renders as nothing at all, which is
        # what the Snipping Tool's own clipboard data would otherwise do.
        opaque = b"\xff" * width
        for row in rows:
            row[3::4] = opaque

    return encode_png(width, height, [bytes(r) for r in rows], 4 if has_alpha else 3)


# ------------------------------------------------------------------ clipboard


def clipboard_available() -> bool:
    """Whether an image can plausibly be read off this machine's clipboard.

    Windows has a first-party API. Linux needs a helper binary -- wl-paste
    under Wayland, xclip under X11 -- so availability there means "one of
    those is installed", which is also the actionable thing to tell someone
    when neither is.
    """
    if sys.platform == "win32":
        return True
    if sys.platform.startswith("linux"):
        import shutil

        return bool(shutil.which("wl-paste") or shutil.which("xclip"))
    return False


def _path_from_uri_list(data: bytes) -> Path | None:
    """First local image path in a text/uri-list payload, if any.

    Copying a file in a Linux file manager puts file:// URIs on the clipboard,
    the moral equivalent of Windows' CF_HDROP. Lines starting with '#' are
    comments per RFC 2483.
    """
    from urllib.parse import unquote, urlparse

    for raw in data.decode("utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parsed = urlparse(line)
        if parsed.scheme != "file":
            continue
        raw_path = unquote(parsed.path)
        # A drive-letter URI (file:///C:/x) parses as "/C:/x" -- the leading
        # slash is a URL artifact, not part of the path. Only ever true on
        # Windows-shaped paths, so stripping it is a no-op everywhere else.
        if len(raw_path) > 2 and raw_path[0] == "/" and raw_path[2] == ":":
            raw_path = raw_path[1:]
        p = Path(raw_path)
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file():
            return p
    return None


def _read_linux_clipboard() -> tuple[str, bytes] | None:
    """Return ("png"|"file", payload) for whatever image the clipboard holds.

    Tries wl-paste (Wayland) before xclip (X11): on a Wayland session xclip
    talks to XWayland's separate clipboard, which is usually stale or empty,
    so the order matters more than it looks.
    """
    import shutil
    import subprocess

    def run(argv: list[str]) -> bytes | None:
        try:
            proc = subprocess.run(argv, capture_output=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return proc.stdout if proc.returncode == 0 else None

    if shutil.which("wl-paste"):
        targets_cmd = ["wl-paste", "--list-types"]
        png_cmd = ["wl-paste", "--type", "image/png"]
        uri_cmd = ["wl-paste", "--type", "text/uri-list"]
    elif shutil.which("xclip"):
        targets_cmd = ["xclip", "-selection", "clipboard", "-o", "-t", "TARGETS"]
        png_cmd = ["xclip", "-selection", "clipboard", "-o", "-t", "image/png"]
        uri_cmd = ["xclip", "-selection", "clipboard", "-o", "-t", "text/uri-list"]
    else:
        raise ImageError(
            "reading an image from the clipboard needs wl-paste (Wayland) or "
            "xclip (X11). Install one: sudo apt install wl-clipboard  (or xclip)"
        )

    targets_raw = run(targets_cmd)
    if targets_raw is None:
        # The tool exists but cannot reach a clipboard -- an SSH session or a
        # server with no display. Distinct from "no image", which is None below.
        raise ImageError(
            "the clipboard is not accessible (no graphical session?)"
        )
    targets = targets_raw.decode("utf-8", errors="replace")

    if "image/png" in targets:
        if data := run(png_cmd):
            return "png", data
    if "text/uri-list" in targets:
        if (data := run(uri_cmd)) is not None:
            if path := _path_from_uri_list(data):
                return "file", str(path).encode("utf-8")
    return None


def _read_windows_clipboard() -> tuple[str, bytes] | None:
    """Return ("png"|"dib"|"file", payload) for whatever image the clipboard holds.

    Formats are tried best-first: a real PNG needs no conversion, a file
    reference needs no decoding at all, and CF_DIB is the fallback that has to
    be re-encoded.
    """
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)

    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
    user32.RegisterClipboardFormatW.restype = wintypes.UINT
    # GlobalLock returns a pointer. Left at the default restype it is truncated
    # to a 32-bit int on 64-bit Python and the read lands nowhere.
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalSize.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalSize.restype = ctypes.c_size_t
    # Prototyped rather than left to ctypes' defaults: the "how many files?"
    # call passes 0xFFFFFFFF, which without a UINT argtype is converted as a
    # signed int.
    shell32.DragQueryFileW.argtypes = [
        wintypes.HANDLE, wintypes.UINT, wintypes.LPWSTR, wintypes.UINT,
    ]
    shell32.DragQueryFileW.restype = wintypes.UINT

    CF_DIB, CF_HDROP, CF_DIBV5 = 8, 15, 17
    cf_png = user32.RegisterClipboardFormatW("PNG")

    # The clipboard is a single global resource; another application can hold
    # it open for a few milliseconds after a copy. Failing the first attempt is
    # normal, so retry briefly before giving up.
    for attempt in range(10):
        if user32.OpenClipboard(None):
            break
        time.sleep(0.02)
    else:
        raise ImageError("another program is holding the clipboard open; try again")

    def payload(fmt: int) -> bytes | None:
        handle = user32.GetClipboardData(fmt)
        if not handle:
            return None
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            return None
        try:
            size = kernel32.GlobalSize(handle)
            return ctypes.string_at(ptr, size)
        finally:
            kernel32.GlobalUnlock(handle)

    try:
        if cf_png and user32.IsClipboardFormatAvailable(cf_png):
            if data := payload(cf_png):
                return "png", data

        if user32.IsClipboardFormatAvailable(CF_HDROP):
            handle = user32.GetClipboardData(CF_HDROP)
            if handle:
                count = shell32.DragQueryFileW(handle, 0xFFFFFFFF, None, 0)
                for i in range(count):
                    buf = ctypes.create_unicode_buffer(1024)
                    shell32.DragQueryFileW(handle, i, buf, 1024)
                    p = Path(buf.value)
                    if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file():
                        return "file", str(p).encode("utf-8")

        for fmt in (CF_DIBV5, CF_DIB):
            if user32.IsClipboardFormatAvailable(fmt):
                if data := payload(fmt):
                    return "dib", data
    finally:
        user32.CloseClipboard()

    return None


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{stem}{suffix}"
    n = 1
    while candidate.exists():
        n += 1
        candidate = directory / f"{stem}-{n}{suffix}"
    return candidate


def grab_clipboard(directory: Path) -> Path | None:
    """Save the clipboard's image into ``directory`` and return its path.

    Returns None when the clipboard holds no image -- the ordinary case of
    hitting the paste key with text on the clipboard, not an error. Raises
    ImageError when there *is* an image but it cannot be converted, because
    then there is something specific to tell the user.
    """
    if sys.platform == "win32":
        found = _read_windows_clipboard()
    elif sys.platform.startswith("linux"):
        found = _read_linux_clipboard()  # raises with an install hint if bare
    else:
        raise ImageError(
            "clipboard images are not supported on this platform yet; attach "
            "the image by path or drag the file into the terminal instead"
        )
    if found is None:
        return None
    kind, payload = found

    if kind == "file":
        # Already a file on disk; attach it where it is rather than making a
        # second copy the user then has to clean up.
        return Path(payload.decode("utf-8"))

    data = payload if kind == "png" else dib_to_png(payload)
    if sniff(data) != "png":
        raise ImageError("the clipboard image did not convert to a valid PNG")

    path = _unique_path(directory, f"pasted-{time.strftime('%Y%m%d_%H%M%S')}", ".png")
    path.write_bytes(data)
    return path
