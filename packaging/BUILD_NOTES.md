# PyInstaller build notes

## Hidden import errors (non-fatal)

If the build log shows:

- **`ERROR: Hidden import 'warp.sim' not found`**  
  The Warp hook was updated to only request warp submodules that actually import. Rebuild with the current `packaging/hooks/hook-warp.py` so optional modules (e.g. `warp.sim`, `warp.render`) are no longer requested.

- **`ERROR: Hidden import 'airclassifier.classification' not found`**  
  There is no top-level `airclassifier.classification` package. If your `.spec` file lists `airclassifier.classification` in `hiddenimports`, remove it. The classification assembly lives under `airclassifier.geometry.assembly.classification`; that module is pulled in by normal imports, so you do not need to add it by name unless you use lazy imports.

Build still completes successfully; these messages only indicate that some requested optional or incorrect modules were not found.
