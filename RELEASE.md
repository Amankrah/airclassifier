# ProteinProcessIO Release Guide

Build and release workflow for the ProteinProcessIO desktop application.

## Prerequisites

### Required Software
- **Python 3.11+** with virtual environment
- **PyInstaller**: `pip install pyinstaller`
- **Inno Setup 6**: Download from https://jrsoftware.org/isinfo.php
- **GitHub CLI**: `winget install GitHub.cli` or https://cli.github.com

### Verify Installation
```powershell
python --version          # Python 3.11+
pyinstaller --version     # PyInstaller 6.x
gh --version              # GitHub CLI 2.x
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" /?  # Inno Setup
```

---

## Quick Release (TL;DR)

```powershell
# 1. Activate venv
.\venv\Scripts\Activate

# 2. Build executable
pyinstaller proteinprocessio.spec --clean --noconfirm

# 3. Test locally
.\dist\ProteinProcessIO\ProteinProcessIO.exe

# 4. Build installer
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss

# 5. Upload to GitHub Releases
gh release upload v1.0.0 "dist\installer\ProteinProcessIO-1.0.0-Setup.exe" --clobber
```

---

## Detailed Workflow

### Step 1: Prepare Release

1. **Update version numbers** in:
   - `installer.iss` → `#define MyAppVersion "X.Y.Z"`
   - `pyproject.toml` → `version = "X.Y.Z"` (if applicable)
   - `src/airclassifier/__init__.py` → `__version__ = "X.Y.Z"` (if appl cable)

2. **Commit all changes**:
   ```powershell
   git add -A
   git commit -m "Release vX.Y.Z"
   git push origin main
   ```

### Step 2: Build PyInstaller Bundle

```powershell
# Activate virtual environment
.\venv\Scripts\Activate

# Clean build (recommended for releases)
pyinstaller proteinprocessio.spec --clean --noconfirm
```

**Build output**: `dist\ProteinProcessIO\`

**Build time**: ~2-5 minutes depending on system

#### Troubleshooting Build Issues

| Issue | Solution |
|-------|----------|
| Missing module errors | Add to `hidden_imports` in `proteinprocessio.spec` |
| Warp JIT fails | Ensure `collect_all('warp')` is in spec file |
| mypyc import errors | Add package to `excludes` list in spec file |
| Large bundle size | Add unused packages to `excludes` list |

### Step 3: Test the Bundle

```powershell
# Run the executable directly
.\dist\ProteinProcessIO\ProteinProcessIO.exe
```

**Verify these work**:
- [ ] Application launches without console errors
- [ ] GPU shows "cuda:0 Ready" (if NVIDIA GPU present)
- [ ] 3D visualization works in Milling page
- [ ] Simulations run correctly
- [ ] All three modules (Pretreatment, Milling, Classification) load

#### Debug Mode

The spec file has `console=True` for debugging. Check console output for errors:
```
GPU: cuda:0 Ready          ✓ Good
GPU: CPU Only              ✓ OK (no NVIDIA GPU)
GPU: Error - ...           ✗ Problem with Warp
PyVista not available...   ✗ Problem with 3D viz
```

### Step 4: Build Windows Installer

```powershell
# Build installer with Inno Setup
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
```

**Output**: `dist\installer\ProteinProcessIO-X.Y.Z-Setup.exe`

**Build time**: ~3-5 minutes (compression is slow)

### Step 5: Test the Installer

1. Run the installer: `.\dist\installer\ProteinProcessIO-X.Y.Z-Setup.exe`
2. Install to a test location
3. Launch from Start Menu shortcut
4. Verify same checks as Step 3

### Step 6: Create GitHub Release

#### First-time release (new version):
```powershell
gh release create vX.Y.Z "dist\installer\ProteinProcessIO-X.Y.Z-Setup.exe" `
  --title "ProteinProcessIO vX.Y.Z" `
  --notes "Release notes here..."
```

#### Update existing release:
```powershell
gh release upload vX.Y.Z "dist\installer\ProteinProcessIO-X.Y.Z-Setup.exe" --clobber
```

#### With release notes from file:
```powershell
gh release create vX.Y.Z "dist\installer\ProteinProcessIO-X.Y.Z-Setup.exe" `
  --title "ProteinProcessIO vX.Y.Z" `
  --notes-file CHANGELOG.md
```

### Step 7: Update Website Download Link

The website at `website/app/download/page.tsx` pulls from GitHub Releases automatically using:
```
https://github.com/anthropics/proteinprocessio/releases/download/vX.Y.Z/ProteinProcessIO-X.Y.Z-Setup.exe
```

Update the version in the download page if needed.

---

## Release Checklist

```
[ ] Version numbers updated
[ ] All changes committed and pushed
[ ] PyInstaller build completed without errors
[ ] Direct executable test passed
[ ] Installer build completed
[ ] Installed app test passed
[ ] GitHub Release created/updated
[ ] Website download link verified
[ ] Release notes written
```

---

## File Reference

| File | Purpose |
|------|---------|
| `proteinprocessio.spec` | PyInstaller configuration |
| `installer.iss` | Inno Setup installer script |
| `run_gui.py` | Application entry point |
| `dist/ProteinProcessIO/` | PyInstaller output |
| `dist/installer/` | Installer output |

---

## Spec File Key Settings

### Hidden Imports
Modules that PyInstaller can't detect automatically:
```python
hidden_imports = [
    'PySide6.QtCore',
    'PySide6.QtCharts',
    'pyvista',
    'pyvistaqt',
    'warp',
    # ... etc
]
```

### Excludes
Dev tools that cause issues in bundles:
```python
excludes = [
    'tkinter',      # Not used
    'black',        # mypyc compiled
    'mypy',         # mypyc compiled
    'mypyc',        # Runtime not bundled
]
```

### Data Files
Files needed at runtime (Warp source for JIT):
```python
datas = [
    # Warp source files collected via collect_all('warp')
    # Application resources
]
```

---

## Common Issues

### "No module named 'xxx'" at runtime
Add the module to `hidden_imports` in the spec file.

### Warp GPU not working
1. Ensure `collect_all('warp')` includes all source files
2. Check that Warp can write to cache directory
3. Verify CUDA toolkit is installed on target machine

### PyVista/3D not working
1. Ensure `collect_all('pyvista')` and `collect_all('vtkmodules')` are in spec
2. Check for mypyc-related import errors (add to excludes)
3. Check console output for specific error message

### Installer too large
Add unused packages to `excludes` list. Common culprits:
- `matplotlib` (if not used)
- `scipy.tests`
- `numpy.tests`

---

## CI/CD Automation (Future)

For GitHub Actions automation, create `.github/workflows/release.yml`:

```yaml
name: Build Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pyinstaller

      - name: Build with PyInstaller
        run: pyinstaller proteinprocessio.spec --clean --noconfirm

      - name: Build Installer
        run: |
          choco install innosetup -y
          & "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss

      - name: Upload to Release
        uses: softprops/action-gh-release@v1
        with:
          files: dist/installer/*.exe
```

---

## Contact

For build issues, check:
1. Console output for specific errors
2. PyInstaller warnings during build
3. GitHub Issues for known problems
