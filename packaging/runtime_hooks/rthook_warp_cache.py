"""Runtime hook: configure Warp kernel cache for frozen (PyInstaller) app.

Warp JIT-compiles GPU kernels and caches them. In a frozen app the default
appdirs path should work, but we set it explicitly to a well-known location
so users can find/clear it if needed.
"""

import os
import sys

if getattr(sys, 'frozen', False):
    cache_base = os.path.join(
        os.path.expanduser('~'), '.cache', 'ProteinProcessIO', 'warp_kernels'
    )
    os.environ.setdefault('WARP_CACHE_PATH', cache_base)
