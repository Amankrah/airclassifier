"""PyInstaller hook for NVIDIA Warp (warp-lang).

Bundles:
- warp/bin/*.so (or .dll on Windows) — native runtime libraries loaded via ctypes
- warp/native/**  — C/CUDA headers required for JIT kernel compilation
- All warp submodules (many are imported dynamically)
"""

import os
from PyInstaller.utils.hooks import collect_submodules

# ---------------------------------------------------------------------------
# Hidden imports — Warp has many internal submodules loaded at runtime
# ---------------------------------------------------------------------------
hiddenimports = collect_submodules('warp')

# ---------------------------------------------------------------------------
# Locate the installed warp package
# ---------------------------------------------------------------------------
import warp
warp_pkg = os.path.dirname(warp.__file__)

binaries = []
datas = []

# ---------------------------------------------------------------------------
# 1. Native shared libraries (warp.so / warp.dll, warp-clang.so / warp-clang.dll)
#    Warp resolves these via:  warp_home = dirname(realpath(warp.__file__))
#                              bin_path  = join(warp_home, "bin")
#    So we must place them at  warp/bin/  inside the frozen bundle.
# ---------------------------------------------------------------------------
bin_dir = os.path.join(warp_pkg, 'bin')
if os.path.isdir(bin_dir):
    for fname in os.listdir(bin_dir):
        fpath = os.path.join(bin_dir, fname)
        if os.path.isfile(fpath):
            binaries.append((fpath, os.path.join('warp', 'bin')))

# ---------------------------------------------------------------------------
# 2. Native C/CUDA headers (needed for JIT kernel compilation at runtime)
#    Warp resolves these via:  inc_path = join(warp_home, "native")
#    We walk the entire native/ tree to capture all subdirectories.
# ---------------------------------------------------------------------------
native_dir = os.path.join(warp_pkg, 'native')
if os.path.isdir(native_dir):
    for root, _dirs, files in os.walk(native_dir):
        for f in files:
            src = os.path.join(root, f)
            rel = os.path.relpath(root, warp_pkg)
            datas.append((src, os.path.join('warp', rel)))
