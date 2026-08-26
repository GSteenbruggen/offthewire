# PyInstaller build for the offline coding agent.
#
#     .venv\Scripts\python.exe -m PyInstaller installer\OffTheWire.spec
#
# Two decisions worth stating, because both are the opposite of the default
# advice you will find:
#
# ONE-DIR, NOT ONE-FILE. A one-file build unpacks the whole bundle into a temp
# directory on every launch and deletes it on exit -- measured at seconds of
# startup for a bundle this size, paid every single time you run the command.
# This is a CLI you invoke constantly from different directories, so that cost
# lands on every invocation. One-dir starts immediately and is what the
# installer copies into Program Files; nobody double-clicks the exe out of a
# download folder, so one-file's single advantage does not apply.
#
# CONSOLE, NOT WINDOWED. It is a terminal program: prompt_toolkit needs a real
# console, and windowed mode would allocate none.
#
# `pathex` carries src/ because the modules import each other flat -- `import
# ui`, `import images` -- rather than as a package. That mirrors the
# sys.path.insert agent.py does at runtime, and is what lets the analysis
# resolve them without restructuring the source into a package.

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []

# trafilatura is imported lazily, inside a function, so static analysis never
# sees it -- and it drags in dateparser and babel, which are data-driven and
# fail at runtime rather than build time when their data is missing. Collect
# each explicitly. Without this the exe builds fine and then cannot read a web
# page, which is the worst kind of packaging bug: invisible until used.
for package in ("trafilatura", "dateparser", "babel", "courlan", "htmldate"):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    except Exception:
        # Optional: --web degrades with a clear message if these are absent.
        continue
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# dateparser builds its language list dynamically; the loaders are reached by
# string lookup and are invisible to the analysis even with collect_all.
hiddenimports += collect_submodules("dateparser.data")

a = Analysis(
    # Forward slashes: PyInstaller accepts them on Windows, and this same spec
    # builds the Linux tarball, where backslashes are just filename characters.
    ["../src/agent.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # The MCP server is a separate entry point with its own dependency, and
    # nothing in the agent imports it. Excluding it keeps the bundle honest
    # about what this executable actually is.
    excludes=["mcp", "tkinter", "test", "unittest", "pydoc_data"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OffTheWire",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX-packed exes are a reliable way to get flagged by AV
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="OffTheWire",
)
