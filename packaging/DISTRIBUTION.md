# ProteinProcessIO v1.0.0 — Distribution & Installation Guide

Desktop application for air classifier design and NVIDIA Warp multiphysics simulation.

---

## System Requirements

### Minimum

| Component | Requirement |
|-----------|-------------|
| OS | Ubuntu 20.04+ / Windows 10+ (64-bit) |
| CPU | x86_64 with SSE4.2 |
| RAM | 8 GB |
| Disk | 1.5 GB free (application + kernel cache) |
| Display | 1280 x 720 |

### Recommended (GPU-accelerated simulation)

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA GPU with CUDA Compute Capability 7.0+ (RTX 20-series or newer) |
| Driver | NVIDIA driver 525+ (CUDA 12.0+) |
| RAM | 16 GB |
| VRAM | 4 GB+ |

The application runs in CPU-only mode if no NVIDIA GPU is detected. Simulations will be slower but fully functional.

---

## Installation

### Linux

1. Download `ProteinProcessIO-1.0.0-linux-x86_64.tar.gz` (387 MB).

2. Extract to your preferred location:
   ```bash
   tar -xzf ProteinProcessIO-1.0.0-linux-x86_64.tar.gz
   ```

3. (Optional) Move to `/opt` for system-wide access:
   ```bash
   sudo mv ProteinProcessIO /opt/ProteinProcessIO
   ```

4. Run the application:
   ```bash
   ./ProteinProcessIO/ProteinProcessIO
   ```

5. (Optional) Create a desktop shortcut:
   ```bash
   cat > ~/.local/share/applications/proteinprocessio.desktop << 'EOF'
   [Desktop Entry]
   Name=ProteinProcessIO
   Comment=Air Classifier Design & Simulation
   Exec=/opt/ProteinProcessIO/ProteinProcessIO
   Type=Application
   Categories=Science;Engineering;
   Terminal=false
   EOF
   ```

### Windows

1. Download `ProteinProcessIO-1.0.0-windows-x86_64.zip`.

2. Extract to your preferred location (e.g. `C:\Program Files\ProteinProcessIO\`).

3. Run `ProteinProcessIO.exe`.

4. (Optional) Right-click `ProteinProcessIO.exe` > **Create shortcut** and move it to your Desktop.

> **Windows SmartScreen:** On first launch Windows may show a "Windows protected your PC" dialog. Click **More info** > **Run anyway**. This is normal for unsigned applications.

---

## First Launch

The first launch takes 10–30 seconds longer than usual. During this time, NVIDIA Warp compiles GPU kernels (JIT compilation) and caches them for future use.

Subsequent launches start immediately using the cached kernels.

### Kernel cache location

| OS | Path |
|----|------|
| Linux | `~/.cache/ProteinProcessIO/warp_kernels/` |
| Windows | `C:\Users\<you>\.cache\ProteinProcessIO\warp_kernels\` |

If you experience issues, delete this directory to force recompilation.

---

## Application Modes

### Air Classification (Ctrl+2)
Full-system or wheel-only particle separation simulation with 3D viewport, real-time KPI tracking, and results analysis.

### RF Pretreatment — GP-15 (Ctrl+1)
Radio-frequency thermal conditioning of whole seeds (yellow pea, faba bean, oat) using the Stalam GP-15 machine model. Includes desirability scoring for thermal treatment, LOX inactivation, protein preservation, moisture retention, and energy efficiency.

---

## Directory Structure

```
ProteinProcessIO/
  ProteinProcessIO          # Main executable (Linux) or .exe (Windows)
  _internal/                # Bundled runtime (do not modify)
    config/
      default_config.yaml   # Default simulation parameters
    warp/
      bin/
        warp.so             # NVIDIA Warp runtime (269 MB)
        warp-clang.so       # LLVM compiler for JIT kernels (63 MB)
      native/               # C/CUDA headers for kernel compilation
    PySide6/                # Qt GUI framework
    ...                     # Other bundled dependencies
```

> **Do not move files out of the `_internal/` directory.** The executable expects this structure to be intact.

---

## Troubleshooting

### Application does not start

**Linux — "Permission denied":**
```bash
chmod +x ProteinProcessIO/ProteinProcessIO
```

**Linux — Missing shared library (libGL, libXcb, etc.):**
```bash
# Ubuntu/Debian
sudo apt install libgl1-mesa-glx libegl1 libxcb-xinerama0 libxkbcommon0

# Fedora
sudo dnf install mesa-libGL mesa-libEGL libxcb libxkbcommon
```

**Windows — "VCRUNTIME140.dll not found":**
Install the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

### 3D viewport is blank or crashes

- Update your GPU driver to the latest version.
- If using a remote desktop or VM, hardware-accelerated OpenGL may not be available. Try setting `MESA_GL_VERSION_OVERRIDE=3.3` before launching.

### "CUDA not available" warning

The application will still run (CPU-only mode). For GPU acceleration:
1. Verify you have an NVIDIA GPU: `nvidia-smi`
2. Ensure driver version is 525+ with CUDA 12.0+ support.
3. The bundled Warp runtime handles CUDA loading — no separate CUDA toolkit installation is required.

### Simulation kernel compilation errors

Delete the kernel cache and restart:
```bash
# Linux
rm -rf ~/.cache/ProteinProcessIO/warp_kernels/

# Windows (PowerShell)
Remove-Item -Recurse ~\.cache\ProteinProcessIO\warp_kernels\
```

### Application settings reset

Settings (window position, last project, preferences) are stored in the OS settings store:
| OS | Location |
|----|----------|
| Linux | `~/.config/AirClassifier/ProteinProcessIO.conf` |
| Windows | Registry: `HKEY_CURRENT_USER\Software\AirClassifier\ProteinProcessIO` |

Delete the above to reset all settings.

---

## Building from Source

### Prerequisites

- Python 3.10+
- NVIDIA GPU + CUDA driver (optional, for GPU mode)

### Steps

```bash
# Clone and set up
git clone <repository-url>
cd airclassifier
python -m venv venv
source venv/bin/activate          # Linux
# venv\Scripts\activate.bat       # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-gui.txt

# Run from source
python run_gui.py

# Build standalone executable
pip install pyinstaller>=6.0
./packaging/build.sh              # Linux
# packaging\build.bat             # Windows
```

Output: `dist/ProteinProcessIO/`

### Build configuration

All packaging configuration lives in `packaging/`:

| File | Purpose |
|------|---------|
| `ProteinProcessIO.spec` | PyInstaller spec (entry point, hidden imports, exclusions) |
| `hooks/hook-warp.py` | Bundles Warp native binaries + JIT headers |
| `hooks/hook-pyvistaqt.py` | Collects PyVista/VTK submodules |
| `runtime_hooks/rthook_warp_cache.py` | Sets kernel cache path for frozen app |
| `build.sh` | Linux build + verification + archive creation |
| `build.bat` | Windows build script |

---

## Uninstallation

Delete the `ProteinProcessIO/` directory and optionally clean up:

```bash
# Linux
rm -rf /opt/ProteinProcessIO                                  # Application
rm -rf ~/.cache/ProteinProcessIO                               # Kernel cache
rm -f ~/.config/AirClassifier/ProteinProcessIO.conf            # Settings
rm -f ~/.local/share/applications/proteinprocessio.desktop      # Desktop shortcut
```

```powershell
# Windows (PowerShell)
Remove-Item -Recurse "C:\Program Files\ProteinProcessIO"       # Application
Remove-Item -Recurse ~\.cache\ProteinProcessIO                 # Kernel cache
# Settings are cleaned from registry automatically if using reg delete
```

---

## Version History

| Version | Date | Notes |
|---------|------|-------|
| 1.0.0 | 2026-02-18 | Initial standalone release. PySide6 GUI with Classification + Pretreatment modes, NVIDIA Warp GPU acceleration, PyVista 3D viewport. |
