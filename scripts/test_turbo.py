"""Tests for turbo mode's decision-making. No GPU, no server, no model.

The launcher itself is exercised live (it needs a real llama-server); what
must be pinned here is everything it decides *before* launching: reading
GGUF metadata correctly, resolving an Ollama model name to its blob through
the manifest tree, and fitting GPU layers to VRAM conservatively.

    .venv\\Scripts\\python.exe scripts\\test_turbo.py
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from turbo import (  # noqa: E402
    TurboError, estimate_gpu_layers, installed_manifest_names, model_facts,
    read_gguf_metadata, resolve_gguf,
)

PASS, FAIL = "  [PASS]", "  [FAIL]"
failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if ok else FAIL} {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures += 1


# ------------------------------------------------------------- GGUF fixtures

def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def make_gguf(path: Path, kvs: list[tuple[str, int, bytes]]) -> None:
    """A minimal valid GGUF: header + metadata, no tensors."""
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
    blob += struct.pack("<Q", len(kvs))
    for key, vtype, payload in kvs:
        blob += _s(key) + struct.pack("<I", vtype) + payload
    path.write_bytes(blob)


def test_gguf_reader() -> None:
    print("\n1. GGUF metadata reader")
    root = Path(tempfile.mkdtemp(prefix="otw-gguf-"))

    f = root / "model.gguf"
    make_gguf(f, [
        ("general.architecture", 8, _s("qwen35")),
        ("qwen35.block_count", 4, struct.pack("<I", 62)),
        ("qwen35.nextn_predict_layers", 4, struct.pack("<I", 1)),
        ("some.flag", 7, struct.pack("<?", True)),
        ("some.floats", 9, struct.pack("<I", 6) + struct.pack("<Q", 2)
         + struct.pack("<f", 1.5) + struct.pack("<f", 2.5)),
        ("tokenizer.huge", 9, struct.pack("<I", 4) + struct.pack("<Q", 100)
         + b"".join(struct.pack("<I", i) for i in range(100))),
    ])

    meta = read_gguf_metadata(f)
    check("strings and ints read", meta.get("general.architecture") == "qwen35"
          and meta.get("qwen35.block_count") == 62)
    check("bools and small arrays read", meta.get("some.flag") is True
          and meta.get("some.floats") == [1.5, 2.5])
    check("huge arrays skipped, not stored", "tokenizer.huge" not in meta)

    facts = model_facts(f)
    check("facts extracted by architecture",
          facts == {"architecture": "qwen35", "block_count": 62, "mtp_layers": 1})

    plain = root / "plain.gguf"
    make_gguf(plain, [
        ("general.architecture", 8, _s("llama")),
        ("llama.block_count", 4, struct.pack("<I", 32)),
    ])
    check("no MTP key means no speculation",
          model_facts(plain)["mtp_layers"] == 0)

    notgguf = root / "bad.bin"
    notgguf.write_bytes(b"MZ\x00\x00whatever")
    try:
        read_gguf_metadata(notgguf)
        check("non-GGUF refused", False, "was read")
    except TurboError:
        check("non-GGUF refused", True)

    trunc = root / "trunc.gguf"
    trunc.write_bytes(f.read_bytes()[:40])
    try:
        read_gguf_metadata(trunc)
        check("truncated file refused", False, "was read")
    except TurboError:
        check("truncated file refused", True)


def test_layer_fitting() -> None:
    print("\n2. GPU layer fitting")

    GB = 1_000_000_000
    # 17 GB model, 62 layers, 16 GB card: must fit *some* but not all.
    n = estimate_gpu_layers(17 * GB, 62, 16 * GB)
    check("big model on small card fits partially", 0 < n < 63, f"{n} layers")
    # tiny model: everything fits, capped at block_count + 1
    check("small model offloads fully",
          estimate_gpu_layers(500_000_000, 28, 16 * GB) == 29)
    check("no VRAM reading means no estimate",
          estimate_gpu_layers(17 * GB, 62, None) is None)
    check("VRAM below headroom means zero layers",
          estimate_gpu_layers(17 * GB, 62, 2 * GB) == 0)
    # the failure asymmetry: the estimate must never exceed what fits
    per_layer = 17 * GB / 63
    check("estimate leaves the headroom intact",
          (n * per_layer) <= 16 * GB - 2_900_000_000, f"{n} layers")


def test_manifest_resolution() -> None:
    print("\n3. Ollama manifest resolution")

    root = Path(tempfile.mkdtemp(prefix="otw-manifests-"))
    blobs = root / "blobs"
    blobs.mkdir(parents=True)

    def install(registry: str, namespace: str, name: str, tag: str,
                digest: str) -> None:
        (blobs / digest.replace(":", "-")).write_bytes(b"GGUFfake")
        mdir = root / "manifests" / registry / namespace / name
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / tag).write_text(json.dumps({"layers": [
            {"mediaType": "application/vnd.docker.container.image.v1+json",
             "digest": "sha256:config"},
            {"mediaType": "application/vnd.ollama.image.model", "digest": digest},
        ]}), encoding="utf-8")

    install("registry.ollama.ai", "library", "qwen3.8", "27b", "sha256:aaa111")
    install("registry.ollama.ai", "orcarouter", "Qwen-Unc", "q4_K_M", "sha256:bbb222")

    p = resolve_gguf("qwen3.8:27b", root)
    check("library model resolves", p.name == "sha256-aaa111", p.name)
    p = resolve_gguf("orcarouter/Qwen-Unc:q4_K_M", root)
    check("namespaced model resolves", p.name == "sha256-bbb222", p.name)

    names = installed_manifest_names(root)
    check("installed listing matches ollama list",
          names == ["orcarouter/Qwen-Unc:q4_K_M", "qwen3.8:27b"], str(names))

    try:
        resolve_gguf("nonexistent:latest", root)
        check("unknown model is a clear error", False, "resolved")
    except TurboError as e:
        check("unknown model is a clear error", "qwen3.8:27b" in str(e), str(e)[:90])

    # a manifest whose blob is gone must say so, not hand back a dead path
    install("registry.ollama.ai", "library", "ghost", "latest", "sha256:ccc333")
    (blobs / "sha256-ccc333").unlink()
    try:
        resolve_gguf("ghost", root)
        check("missing blob is a clear error", False, "resolved")
    except TurboError as e:
        check("missing blob is a clear error", "missing" in str(e), str(e)[:90])


def main() -> int:
    print("=" * 68)
    print("TURBO MODE TESTS")
    print("=" * 68)
    test_gguf_reader()
    test_layer_fitting()
    test_manifest_resolution()
    print("\n" + "=" * 68)
    print("All turbo tests passed." if not failures else f"{failures} FAILED.")
    print("=" * 68)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
