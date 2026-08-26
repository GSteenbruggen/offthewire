"""Tests for attaching images to a turn. No network, no model.

The three ways an image gets into a terminal -- clipboard, drag-and-drop, a
typed path -- all converge on a file path, and everything here is about that
path surviving intact from the keystroke to the wire and back off disk.

The clipboard tests build DIBs by hand rather than requiring something to be
copied first, so this runs headless. The one live clipboard read at the end is
reported, not asserted: whether an image happens to be on the clipboard is not
a property of this code.

    .venv\\Scripts\\python.exe scripts\\test_images.py
"""

from __future__ import annotations

import base64
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import images as IM  # noqa: E402
import session as S  # noqa: E402
import transcript as T  # noqa: E402

PASS, FAIL = "  [PASS]", "  [FAIL]"
failures = 0

TMP = Path(tempfile.mkdtemp(prefix="agent-images-"))


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if ok else FAIL} {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures += 1


def dib(width: int, height: int, bits: int, pixels: bytes) -> bytes:
    """A BITMAPINFOHEADER DIB, the shape CF_DIB actually arrives in."""
    header = struct.pack(
        "<IiiHHIIiiII", 40, width, height, 1, bits, 0, 0, 0, 0, 0, 0
    )
    return header + pixels


def sample_png() -> bytes:
    # 4x3, solid red, three channels
    return IM.encode_png(4, 3, [bytes([255, 0, 0] * 4)] * 3, 3)


def test_formats() -> None:
    print("\n1. Identifying and loading files")

    png = sample_png()
    path = TMP / "red.png"
    path.write_bytes(png)

    check("PNG recognised by content", IM.sniff(png) == "png", IM.sniff(png))
    check("dimensions read", IM.dimensions(png, "png") == (4, 3))

    att = IM.load(path)
    check("loads", att.kind == "png" and (att.width, att.height) == (4, 3), att.summary)
    check("base64 round trips", base64.b64decode(att.b64) == png)

    # Extension is never trusted: a browser saving a WEBP as .png is common,
    # and Ollama's failure for one is opaque.
    liar = TMP / "liar.png"
    liar.write_text("this is prose")
    try:
        IM.load(liar)
        check("content decides, not the extension", False)
    except IM.ImageError as e:
        check("content decides, not the extension", True, str(e))

    bmp = TMP / "shot.bmp"
    bmp.write_bytes(b"BM" + b"\x00" * 60)
    try:
        IM.load(bmp)
        check("unsupported format refused with advice", False)
    except IM.ImageError as e:
        check("unsupported format refused with advice", "PNG or JPEG" in str(e), str(e))

    try:
        IM.load(TMP / "nope.png")
        check("missing file refused", False)
    except IM.ImageError:
        check("missing file refused", True)


def test_dib_conversion() -> None:
    """CF_DIB is a raw buffer: bottom-up, BGR, rows padded to four bytes.

    Every one of those is a way to produce a PNG that is upside down, blue
    where it should be red, or sheared diagonally.
    """
    print("\n2. Clipboard bitmap conversion")

    # 2x2, 32bpp, bottom-up. Written bottom row first, as Windows stores it.
    bottom = bytes([0, 0, 255, 0]) + bytes([0, 255, 0, 0])  # red, green
    top = bytes([255, 0, 0, 0]) + bytes([0, 0, 0, 0])  # blue, black
    png = IM.dib_to_png(dib(2, 2, 32, bottom + top))
    check("converts to a valid PNG", IM.sniff(png) == "png")
    check("size preserved", IM.dimensions(png, "png") == (2, 2))

    px = decode_png_pixels(png, 2, 2, 4)
    check("top-left is blue (rows flipped)", px(0, 0) == (0, 0, 255, 255), str(px(0, 0)))
    check("bottom-left is red (BGR swapped)", px(0, 1) == (255, 0, 0, 255), str(px(0, 1)))
    check(
        "zero alpha treated as opaque",
        all(px(x, y)[3] == 255 for x in range(2) for y in range(2)),
    )

    # 24bpp, 1px wide: 3 bytes of pixel plus 1 byte of row padding.
    png24 = IM.dib_to_png(dib(1, 2, 24, b"\x10\x20\x30\x00" + b"\x40\x50\x60\x00"))
    check("row padding skipped", IM.dimensions(png24, "png") == (1, 2))
    px24 = decode_png_pixels(png24, 1, 2, 3)
    check("24bpp channels ordered", px24(0, 1) == (0x30, 0x20, 0x10), str(px24(0, 1)))

    try:
        IM.dib_to_png(dib(2, 2, 8, b"\x00" * 16))
        check("palette bitmaps refused rather than guessed at", False)
    except IM.ImageError as e:
        check("palette bitmaps refused rather than guessed at", True, str(e))

    try:
        IM.dib_to_png(b"\x00" * 10)
        check("truncated buffer refused", False)
    except IM.ImageError:
        check("truncated buffer refused", True)


def decode_png_pixels(png: bytes, width: int, height: int, channels: int):
    """Read pixels back out of our own PNG, to prove the encoder is honest."""
    import zlib

    idat, i = b"", 8
    while i < len(png):
        length = struct.unpack(">I", png[i : i + 4])[0]
        if png[i + 4 : i + 8] == b"IDAT":
            idat += png[i + 8 : i + 8 + length]
        i += 12 + length
    raw = zlib.decompress(idat)
    stride = 1 + width * channels  # leading filter byte per row

    def at(x: int, y: int) -> tuple:
        off = y * stride + 1 + x * channels
        return tuple(raw[off : off + channels])

    return at


def test_extraction() -> None:
    print("\n3. Finding image references in a typed line")

    png = sample_png()
    spaced = TMP / "screen shot.png"
    spaced.write_bytes(png)
    plain = TMP / "b.png"
    plain.write_bytes(png)

    e = IM.extract(f"what is wrong here? {IM.marker(spaced)}")
    check("marker recognised", e.refs == [spaced], str(e.refs))
    check(
        "marker becomes a label when the image is sent",
        e.render(e.refs) == "what is wrong here? [image 1: screen shot.png]",
        repr(e.render(e.refs)),
    )
    check(
        "marker falls back to the path when it is not",
        str(spaced) in e.render([]),
        repr(e.render([])),
    )

    # How a drag-and-drop and a shell completion both arrive.
    e = IM.extract(f'compare "{spaced}" and {plain} please')
    check("quoted and bare paths both found", e.refs == [spaced, plain], str(e.refs))
    check(
        "numbered in the order they appear",
        e.render(e.refs)
        == "compare [image 1: screen shot.png] and [image 2: b.png] please",
        repr(e.render(e.refs)),
    )

    # The safe direction: text that merely looks like a path stays text.
    e = IM.extract("the bug is in logo.png somewhere")
    check("a path that does not exist stays prose", not e.has_images)
    check("and the line is untouched", e.render() == "the bug is in logo.png somewhere")

    e = IM.extract("read tools.py and fix it")
    check("non-image files ignored", not e.has_images)

    e = IM.extract(f"{plain} and {plain} again")
    check("the same file is sent once", e.refs == [plain], str(e.refs))
    check(
        "but labelled at both mentions",
        e.render(e.refs) == "[image 1: b.png] and [image 1: b.png] again",
        repr(e.render(e.refs)),
    )

    e = IM.extract(f"look at {plain}.")
    check("trailing punctuation is not part of the path", e.refs == [plain])

    e = IM.extract("b.png", base=TMP)
    check("relative paths resolve against the workspace", e.refs == [TMP / "b.png"])

    # The numbering must describe what was actually attached, not what was
    # mentioned -- otherwise a failed load leaves the model reading about an
    # "[image 2]" that was never sent.
    broken = TMP / "liar.png"
    e = IM.extract(f"{plain} and {broken} together")
    loaded, errors = IM.load_all(e.refs)
    check("one loads, one reports", len(loaded) == 1 and len(errors) == 1, str(errors))
    check(
        "labels renumber around the failure",
        e.render([a.path for a in loaded]) == f"[image 1: b.png] and {broken} together",
        repr(e.render([a.path for a in loaded])),
    )


def test_persistence() -> None:
    """Base64 must never reach the session file.

    A screenshot is a couple of megabytes; inline it and every session file
    holding one is unreadable and slow to resume for the rest of its life.
    """
    print("\n4. Sessions store paths, not pixels")

    png = sample_png()
    path = TMP / "kept.png"
    path.write_bytes(png)
    att = IM.load(path)

    sdir = TMP / "sessions"
    sess = S.Session(
        model="m", context_limit=8192, system_prompt="sys", sessions_dir=sdir,
        persist=True,
    )
    before = sess.used_tokens()
    sess.append(
        {
            "role": "user",
            "content": "[image 1: kept.png]",
            "images": [att.b64],
            "image_paths": [str(path)],
        }
    )

    on_disk = sess.path.read_text(encoding="utf-8")
    check("no base64 written", att.b64[:32] not in on_disk, f"{len(on_disk)} chars")
    check("path written", "kept.png" in on_disk)
    check(
        "image counted against the context budget",
        sess.used_tokens() - before > IM.TOKENS_PER_IMAGE,
    )

    wire = sess.wire_messages()[-1]
    check("images go on the wire", wire.get("images") == [att.b64])
    check("bookkeeping does not", "image_paths" not in wire)
    check("the live message keeps both", "image_paths" in sess.messages[-1])

    back = S.Session.resume(
        sess.path, model="m", context_limit=8192, system_prompt="sys", persist=True
    )
    check("resume re-reads the file", back.messages[-1].get("images") == [att.b64])
    check("nothing reported missing", back.missing_images == [])

    # A model asked about a picture it cannot see will describe one anyway, so
    # a deleted attachment has to be said out loud rather than quietly dropped.
    path.unlink()
    gone = S.Session.resume(
        sess.path, model="m", context_limit=8192, system_prompt="sys", persist=True
    )
    check("a deleted attachment is reported", len(gone.missing_images) == 1)
    check(
        "and the text stops claiming it is there",
        "no longer available" in gone.messages[-1]["content"],
    )
    check("with no empty images key left behind", "images" not in gone.messages[-1])

    rendered = T.render(T.load(sess.path), colour=False)
    check("transcript shows the attachment", "attached: kept.png" in rendered)
    check(
        "transcript stays ASCII for a cp1252 console",
        rendered.isascii(),
        repr([c for c in rendered if not c.isascii()][:5]),
    )


def test_not_saved_by_default() -> None:
    """A conversation leaves no trace unless asked to.

    The whole point is that it is the *default*, so the thing worth asserting
    is not that append() skips a write -- it is that the directory is still not
    there afterwards. An earlier version created it eagerly in __post_init__
    and left an empty folder behind on every run.
    """
    print("\n5. Nothing is written unless --save")

    sdir = TMP / "unsaved" / "sessions"
    sess = S.Session(
        model="m", context_limit=8192, system_prompt="sys", sessions_dir=sdir
    )
    check("persist is off unless asked", sess.persist is False)

    sess.append({"role": "user", "content": "something private"})
    sess.append({"role": "assistant", "content": "a reply"})

    check("no session file", not sess.path.exists())
    check("no sessions directory at all", not sdir.exists(), str(sdir))
    check("the conversation is still in memory", len(sess.messages) == 2)
    check("token accounting still works", sess.used_tokens() > 0)

    # Compaction writes a marker line; it must not resurrect the file either.
    sess._write_line({"role": "_system_note", "content": "compacted"})
    check("internal writes are gated too", not sess.path.exists())

    # And the opposite, in the same shape, so the gate is provably two-way.
    saved = S.Session(
        model="m", context_limit=8192, system_prompt="sys",
        sessions_dir=TMP / "saved" / "sessions", persist=True,
    )
    saved.append({"role": "user", "content": "keep me"})
    check("--save still writes", saved.path.exists())
    check("and the content is there", "keep me" in saved.path.read_text(encoding="utf-8"))


def test_uri_list() -> None:
    """text/uri-list is Linux's CF_HDROP: file managers put file:// URIs on
    the clipboard when a file is copied. Pure parsing, testable anywhere."""
    print("\n6. URI-list parsing (Linux clipboard file-copy)")

    img = TMP / "uri target.png"
    img.write_bytes(sample_png())
    uri = "file://" + str(img).replace("\\", "/").replace(" ", "%20")
    if not uri.startswith("file:///"):
        uri = uri.replace("file://", "file:///")

    found = IM._path_from_uri_list(f"# comment line\r\n{uri}\r\n".encode())
    check("file URI with escaped space resolves", found == img, str(found))
    check("comments are skipped", IM._path_from_uri_list(b"# only a comment\n") is None)
    check(
        "non-file schemes are ignored",
        IM._path_from_uri_list(b"https://example.com/a.png\n") is None,  # offline-fixture
    )
    check(
        "non-image files are ignored",
        IM._path_from_uri_list(b"file:///etc/hostname\n") is None,
    )
    missing = "file:///" + str(TMP / "no-such.png").replace("\\", "/")
    check("missing files are ignored", IM._path_from_uri_list(missing.encode()) is None)


def test_temp_pastes_do_not_outlive_the_turn() -> None:
    """Without --save, a clipboard paste is disk residue the moment its bytes
    are in memory. The product promise is 'nothing on disk unless you ask',
    and a screenshot in %TEMP% breaks it as surely as a transcript would."""
    from agent import Agent
    from ollama_client import OllamaClient
    from tools import Workspace

    print("\n7. Temp pastes do not outlive the turn")

    ws = Workspace(str(TMP))
    client = OllamaClient()

    # --- save OFF: the pasted file is consumed, then removed
    agent = Agent(client, "m", ws, interactive=False, save=False, supports_vision=True)
    agent.images_dir.mkdir(parents=True, exist_ok=True)
    pasted = agent.images_dir / "pasted-test.png"
    pasted.write_bytes(sample_png())

    # A file the user referenced by path, sitting anywhere else, is theirs.
    user_file = TMP / "users own.png"
    user_file.write_bytes(sample_png())

    text, atts = agent._attach_images(f'{IM.marker(pasted)} versus "{user_file}"')
    check("both images attached", len(atts) == 2, text)
    check("bytes are in memory", all(len(a.data) > 0 for a in atts))
    check("pasted temp file removed", not pasted.exists())
    check("user's own file untouched", user_file.exists())

    # An orphaned paste (marker edited out, never attached) goes at exit.
    orphan = agent.images_dir / "pasted-orphan.png"
    orphan.write_bytes(sample_png())
    agent.cleanup_temp_pastes()
    check("orphaned paste swept on exit", not orphan.exists())

    # --- save ON: pastes are part of the record and must survive
    saver = Agent(client, "m", ws, interactive=False, save=True, supports_vision=True)
    saver.session.sessions_dir = TMP / "kept-sessions"
    kept_dir = saver.images_dir
    kept_dir.mkdir(parents=True, exist_ok=True)
    kept = kept_dir / "pasted-keep.png"
    kept.write_bytes(sample_png())
    saver._attach_images(IM.marker(kept))
    check("with --save the paste is kept", kept.exists())
    saver.cleanup_temp_pastes()
    check("exit sweep never touches saved pastes", kept.exists())


def test_clipboard() -> None:
    print("\n8. Clipboard")
    if sys.platform == "win32":
        check("available on Windows", IM.clipboard_available() is True)
    elif sys.platform.startswith("linux"):
        import shutil

        helpers = bool(shutil.which("wl-paste") or shutil.which("xclip"))
        check(
            "availability tracks helper binaries",
            IM.clipboard_available() == helpers,
            f"helpers={helpers}",
        )
    elif sys.platform == "darwin":
        check("available on macOS (osascript ships with the OS)",
              IM.clipboard_available() is True)
        _macos_round_trip()
    else:
        check("unsupported platforms say so", IM.clipboard_available() is False)


def _macos_round_trip() -> None:
    """Put a known PNG on the real pasteboard, read it back through our code.

    This is the one place the AppleScript hex-dump parsing is actually
    exercised, and it needs real macOS -- which in this project means the CI
    runner. A pasteboard that cannot be written (some hardened CI images) is
    reported and skipped rather than failed: that is a property of the
    machine, not of this code.
    """
    import subprocess

    png_file = TMP / "clip source.png"
    png_file.write_bytes(sample_png())
    posix = str(png_file)

    setter = subprocess.run(
        ["osascript", "-e",
         f'set the clipboard to (read (POSIX file "{posix}") as «class PNGf»)'],
        capture_output=True, text=True, timeout=15,
    )
    if setter.returncode != 0:
        print(f"       (pasteboard not writable here; skipping round-trip: "
              f"{setter.stderr.strip()[:80]})")
        return

    got = IM.grab_clipboard(TMP / "pasted")
    check("round-trip returns a file", got is not None, str(got))
    if got is not None:
        data = got.read_bytes()
        check("round-trip payload is PNG", IM.sniff(data) == "png")
        check(
            "round-trip pixels survive",
            IM.dimensions(data, "png") == (4, 3),
            str(IM.dimensions(data, "png")),
        )
    if not IM.clipboard_available():
        print("       (not Windows -- clipboard paste is unavailable by design)")
        return
    try:
        got = IM.grab_clipboard(TMP / "pasted")
    except IM.ImageError as e:
        print(f"       live read: {e}")
        return
    # Not asserted: what is on the clipboard is not a property of this code.
    print(f"       live read: {got if got else 'no image on the clipboard'}")
    if got:
        check("what was grabbed is loadable", IM.load(got).kind in IM.SUPPORTED)


def main() -> int:
    print("=" * 70)
    print("IMAGE ATTACHMENT TESTS")
    print("=" * 70)

    test_formats()
    test_dib_conversion()
    test_extraction()
    test_persistence()
    test_not_saved_by_default()
    test_uri_list()
    test_temp_pastes_do_not_outlive_the_turn()
    test_clipboard()

    print("\n" + "=" * 70)
    print("All image tests passed." if not failures else f"{failures} FAILED.")
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
