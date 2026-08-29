"""One-command llama-server launch: ``OffTheWire --turbo``.

The manual runbook this replaces had five steps -- free the GPU, set two
CUDA environment variables, resolve a model name to its GGUF blob, launch
``llama-server`` with eight flags, wait for it to come up -- and getting the
environment lines wrong produces a server that silently runs on CPU at a
quarter of the speed. Mechanical sequences with sharp edges belong in code.

What this module knows how to do:

  * find the ``llama-server`` binary Ollama ships (no separate install)
  * resolve an Ollama model name to its GGUF file by reading Ollama's
    manifest files directly -- works even when Ollama is not running
  * read GGUF metadata (small, stable binary format) to learn the layer
    count -- for fitting layers to measured VRAM -- and whether the model
    carries an MTP draft head, which decides speculative decoding
  * launch the server as a child process with the CUDA paths set, wait for
    health, and kill the whole tree at exit so the VRAM comes back

Everything is loopback: the server binds a local port and the agent
connects through the same guard as every other backend.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import paths

# Reserve for the KV cache, compute buffers and the desktop compositor when
# fitting layers. Deliberately conservative: a too-low -ngl costs a little
# speed, a too-high one crashes the server at load.
TURBO_VRAM_HEADROOM = 3_000_000_000

# Ports tried in order. 8080 is deliberately absent -- the SearXNG container
# owns it on machines with web lookup installed.
CANDIDATE_PORTS = [8081, 8082, 8083, 8084, 8085]

HEALTH_TIMEOUT_S = 300  # a 17 GB model takes a while to read off disk


class TurboError(RuntimeError):
    """Anything that should stop startup with a sentence, not a traceback."""


# --------------------------------------------------------------- GGUF reading

_SIMPLE_TYPES = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}


def read_gguf_metadata(path: Path, max_array: int = 64) -> dict[str, Any]:
    """The key/value metadata block of a GGUF file, as a dict.

    The format is small and stable (magic, version, counts, then typed
    key/values), which is why eighty lines here beat a dependency. Arrays
    larger than ``max_array`` (tokenizer vocabularies run to 150k entries)
    are skipped over but not kept.
    """
    with open(path, "rb") as f:

        def take(n: int) -> bytes:
            b = f.read(n)
            if len(b) < n:
                raise TurboError(f"{path.name} is truncated; not a valid GGUF file")
            return b

        if take(4) != b"GGUF":
            raise TurboError(f"{path.name} is not a GGUF file")
        version = struct.unpack("<I", take(4))[0]
        if version < 2:
            raise TurboError(f"GGUF version {version} is too old to read")
        take(8)  # tensor count -- not needed
        n_kv = struct.unpack("<Q", take(8))[0]

        def read_string() -> str:
            n = struct.unpack("<Q", take(8))[0]
            return take(n).decode("utf-8", "replace")

        def read_value(vtype: int) -> Any:
            if vtype in _SIMPLE_TYPES:
                fmt, size = _SIMPLE_TYPES[vtype]
                return struct.unpack(fmt, take(size))[0]
            if vtype == 8:
                return read_string()
            if vtype == 9:
                etype = struct.unpack("<I", take(4))[0]
                count = struct.unpack("<Q", take(8))[0]
                values = [read_value(etype) for _ in range(count)]
                return values if count <= max_array else None
            raise TurboError(f"unknown GGUF value type {vtype} in {path.name}")

        meta: dict[str, Any] = {}
        for _ in range(n_kv):
            key = read_string()
            vtype = struct.unpack("<I", take(4))[0]
            value = read_value(vtype)
            if value is not None:
                meta[key] = value
        return meta


def model_facts(gguf: Path) -> dict[str, Any]:
    """The three metadata facts turbo mode decides by."""
    meta = read_gguf_metadata(gguf)
    arch = meta.get("general.architecture", "")
    return {
        "architecture": arch,
        "block_count": meta.get(f"{arch}.block_count", 0),
        "mtp_layers": meta.get(f"{arch}.nextn_predict_layers", 0),
    }


# ------------------------------------------------------------ Ollama lookups

def ollama_models_root() -> Path:
    if override := os.environ.get("OLLAMA_MODELS"):
        return Path(override)
    return Path.home() / ".ollama" / "models"


def installed_manifest_names(root: Path | None = None) -> list[str]:
    """Every model:tag installed locally, straight from the manifest tree."""
    manifests = (root or ollama_models_root()) / "manifests"
    names = []
    for tag_file in manifests.glob("*/*/*/*"):
        if tag_file.is_file():
            name = tag_file.parent.name
            namespace = tag_file.parent.parent.name
            prefix = "" if namespace == "library" else f"{namespace}/"
            names.append(f"{prefix}{name}:{tag_file.name}")
    return sorted(names)


def resolve_gguf(model: str, root: Path | None = None) -> Path:
    """An Ollama model name to the GGUF blob on disk, no Ollama required.

    A manifest is a small JSON file at ``manifests/<registry>/<namespace>/
    <name>/<tag>`` whose model layer names a blob by digest. Reading it
    directly means --turbo works when Ollama is stopped -- which it usually
    is, since turbo exists to run the model *instead* of Ollama.
    """
    root = root or ollama_models_root()
    name, _, tag = model.partition(":")
    tag = tag or "latest"
    patterns = (
        [f"*/{name}/{tag}"] if "/" in name else [f"*/library/{name}/{tag}", f"*/*/{name}/{tag}"]
    )
    manifest = None
    for pattern in patterns:
        found = sorted((root / "manifests").glob(pattern))
        if found:
            manifest = found[0]
            break
    if manifest is None:
        installed = installed_manifest_names(root)
        listing = ("\n  installed: " + ", ".join(installed)) if installed else ""
        raise TurboError(f"no installed Ollama model named {model!r}.{listing}")

    try:
        layers = json.loads(manifest.read_text(encoding="utf-8")).get("layers", [])
    except (OSError, ValueError) as e:
        raise TurboError(f"could not read the manifest for {model!r}: {e}")
    for layer in layers:
        if str(layer.get("mediaType", "")).endswith("image.model"):
            blob = root / "blobs" / str(layer.get("digest", "")).replace(":", "-")
            if blob.is_file():
                return blob
            raise TurboError(
                f"the manifest for {model!r} names a blob that is missing: {blob}"
            )
    raise TurboError(f"the manifest for {model!r} has no model layer")


def find_llama_server() -> Path:
    """The llama-server binary Ollama ships, or one on PATH."""
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    candidates = []
    if os.name == "nt" and (base := os.environ.get("LOCALAPPDATA")):
        candidates.append(Path(base) / "Programs" / "Ollama" / "lib" / "ollama" / exe)
    candidates += [
        Path("/usr/local/lib/ollama") / exe,
        Path("/usr/lib/ollama") / exe,
        Path("/opt/homebrew/lib/ollama") / exe,
    ]
    for c in candidates:
        if c.is_file():
            return c
    import shutil

    if which := shutil.which(exe):
        return Path(which)
    raise TurboError(
        "could not find llama-server. Ollama ships it (reinstalling Ollama "
        "restores it), or install llama.cpp and put llama-server on PATH."
    )


def cuda_environment(server: Path) -> dict[str, str]:
    """The environment that makes llama-server actually use the GPU.

    Two facts a whole day of debugging established: PATH must contain the
    cuda directory so the CUDA runtime DLLs resolve, and GGML_BACKEND_PATH
    must point at the ggml-cuda library *file*, not its directory --
    LoadLibrary on a directory fails and the server silently runs on CPU at
    a quarter of the speed.
    """
    env = dict(os.environ)
    lib_dir = server.parent
    dll_name = "ggml-cuda.dll" if os.name == "nt" else "libggml-cuda.so"
    best: Path | None = None
    for sub in sorted(lib_dir.glob("cuda_v*"), reverse=True):  # prefer newest
        if (sub / dll_name).is_file():
            best = sub
            break
    if best is None and (lib_dir / dll_name).is_file():
        best = lib_dir
    if best is not None:
        sep = ";" if os.name == "nt" else ":"
        env["PATH"] = f"{best}{sep}" + env.get("PATH", "")
        env["GGML_BACKEND_PATH"] = str(best / dll_name)
    return env


def estimate_gpu_layers(
    model_bytes: int, block_count: int, vram_bytes: int | None,
    headroom: int = TURBO_VRAM_HEADROOM,
) -> int | None:
    """How many layers fit in measured VRAM; None when VRAM is unknown.

    Approximation on purpose: layers are treated as equally sized (the
    output layer counted as one more). Conservative headroom, because the
    failure modes are asymmetric -- a few layers too few is slightly slower,
    one too many is a crash at load.
    """
    if not vram_bytes or not block_count or not model_bytes:
        return None
    per_layer = model_bytes / (block_count + 1)
    usable = vram_bytes - headroom
    if usable <= 0:
        return 0
    return min(block_count + 1, int(usable / per_layer))


# ----------------------------------------------------------------- launching

async def _port_is_free(port: int) -> bool:
    """A port is free when nothing answers HTTP there. Loopback only.

    Both refusal shapes count as free: a closed port answers with an
    instant refusal on most machines, but Windows setups exist (observed on
    this project's own reference machine) where the connection is dropped
    and times out instead. Anything that actually answers -- any status,
    any protocol error after connecting -- is occupied.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=1.5) as c:
            await c.get(f"http://127.0.0.1:{port}/")
        return False  # something answered
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return True
    except Exception:
        return False  # answered strangely -- still occupied


async def free_port() -> int:
    for port in CANDIDATE_PORTS:
        if await _port_is_free(port):
            return port
    raise TurboError(
        f"no free port among {CANDIDATE_PORTS}; stop something or pass --host"
    )


def free_vram_bytes() -> int | None:
    """Currently unused VRAM across NVIDIA GPUs, or None when unreadable."""
    from models import parse_vram_readings

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return parse_vram_readings(proc.stdout)


async def free_the_gpu(needed_bytes: int, note: Callable[[str], None]) -> bool:
    """Ask Ollama to evict resident models -- but only when it must.

    Evicting is reversible (Ollama reloads on next use) but not free: it
    interrupts whatever else is using that model. So a target that already
    fits in the VRAM left over is loaded alongside instead. Best-effort by
    design: Ollama not running is the normal case, not an error.

    Returns whether anything was evicted, because the layer estimate should
    then budget against total VRAM rather than the pre-eviction free figure.
    """
    free = free_vram_bytes()
    if free is not None and needed_bytes + TURBO_VRAM_HEADROOM <= free:
        return False

    from ollama_client import OllamaClient

    evicted = False
    try:
        client = OllamaClient()
        if not await client.ping():
            return False
        for m in await client.ps():
            name = m.get("model") or m.get("name", "")
            if name:
                note(f"unloading {name} from Ollama to free the GPU…")
                await client.unload(name)
                evicted = True
    except Exception:
        pass
    return evicted


@dataclass
class TurboServer:
    host: str
    model_name: str
    gpu_layers: int | None
    speculative: bool
    log_path: Path
    process: Any = field(default=None, repr=False)

    def stop(self) -> None:
        """Kill the server and its whole tree; the VRAM comes back."""
        proc = self.process
        if proc is None or proc.returncode is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                )
            else:
                proc.terminate()
        except Exception:
            pass


def _log_tail(log_path: Path, lines: int = 12) -> str:
    try:
        return "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        )
    except OSError:
        return "(no log)"


async def launch_turbo(
    model: str,
    *,
    context: int = 32768,
    gpu_layers: int | None = None,
    note: Callable[[str], None] = print,
) -> TurboServer:
    """Resolve, configure, launch, and wait for health. The whole runbook."""
    server = find_llama_server()
    gguf = resolve_gguf(model)
    facts = model_facts(gguf)
    gguf_bytes = gguf.stat().st_size
    speculative = facts["mtp_layers"] >= 1
    port = await free_port()
    evicted = await free_the_gpu(gguf_bytes, note)

    if gpu_layers is None:
        from models import total_vram_bytes

        # After an eviction the freed memory takes a moment to show up in
        # nvidia-smi, so budget against the whole card; otherwise fit into
        # what is actually free alongside whoever else is resident.
        vram = total_vram_bytes() if evicted else (free_vram_bytes() or total_vram_bytes())
        gpu_layers = estimate_gpu_layers(gguf_bytes, facts["block_count"], vram)

    cmd = [
        str(server), "-m", str(gguf),
        "--jinja",
        "-c", str(context),
        "-fa", "on",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--reasoning-format", "deepseek",
        "--port", str(port),
        "--host", "127.0.0.1",
    ]
    if gpu_layers is not None:
        cmd += ["-ngl", str(gpu_layers)]
    if speculative:
        cmd += ["--spec-type", "draft-mtp"]

    log_dir = paths.data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / time.strftime("llama-server-%Y%m%d-%H%M%S.log")

    if gpu_layers is None:
        layers_note = "GPU split left to the server (no VRAM reading)"
    elif facts["block_count"] and gpu_layers > facts["block_count"]:
        layers_note = "all layers on GPU"
    else:
        layers_note = f"{gpu_layers}/{facts['block_count'] or '?'} layers on GPU"
    note(
        f"turbo: {gguf.name[:19]}… · {layers_note} · "
        f"speculative decoding {'on (MTP head)' if speculative else 'off (no MTP head)'} · "
        f"port {port}"
    )
    note(f"server log: {log_path}")

    log_file = open(log_path, "w", encoding="utf-8", errors="replace")
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=cuda_environment(server),
            cwd=str(server.parent),
        )
    except OSError as e:
        log_file.close()
        raise TurboError(f"could not start llama-server: {e}")

    result = TurboServer(
        host=f"localhost:{port}",
        model_name=model,
        gpu_layers=gpu_layers,
        speculative=speculative,
        log_path=log_path,
        process=process,
    )

    from openai_compat import OpenAICompatClient

    probe = OpenAICompatClient("llamacpp", result.host)
    note("loading the model…")
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.returncode is not None:
            raise TurboError(
                "llama-server exited during load. Last log lines:\n"
                + _log_tail(log_path)
                + (
                    "\n(an out-of-memory error here usually means the GPU "
                    "layer estimate was too high -- retry with --gpu-layers "
                    f"{max(0, (gpu_layers or 8) - 8)})"
                    if gpu_layers
                    else ""
                )
            )
        if await probe.ping():
            return result
        await asyncio.sleep(1.0)

    result.stop()
    raise TurboError(
        f"llama-server did not become healthy within {HEALTH_TIMEOUT_S}s. "
        f"Last log lines:\n" + _log_tail(log_path)
    )
