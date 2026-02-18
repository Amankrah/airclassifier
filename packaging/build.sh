#!/usr/bin/env bash
# ProteinProcessIO — Linux build script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DIST_DIR="$PROJECT_ROOT/dist"
BUILD_DIR="$PROJECT_ROOT/build"
VENV="$PROJECT_ROOT/venv"
APP_NAME="ProteinProcessIO"
VERSION="1.0.0"

echo "=== $APP_NAME Linux Build ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Activate venv
source "$VENV/bin/activate"

# Ensure PyInstaller is installed
pip install --quiet "pyinstaller>=6.0"

# Clean previous build artifacts
rm -rf "$BUILD_DIR/$APP_NAME" "$DIST_DIR/$APP_NAME"

echo "Running PyInstaller..."
pyinstaller "$SCRIPT_DIR/$APP_NAME.spec" \
    --distpath "$DIST_DIR" \
    --workpath "$BUILD_DIR" \
    --noconfirm

# ── Verify build ──────────────────────────────────────────────
echo ""
echo "=== Verifying build ==="
BUNDLE="$DIST_DIR/$APP_NAME"
FAIL=0

check_file() {
    if [ -f "$1" ]; then
        echo "  OK: $(basename "$1") ($(du -sh "$1" | cut -f1))"
    else
        echo "  MISSING: $1"
        FAIL=1
    fi
}

check_file "$BUNDLE/$APP_NAME"
check_file "$BUNDLE/_internal/warp/bin/warp.so"
check_file "$BUNDLE/_internal/warp/bin/warp-clang.so"

if [ -d "$BUNDLE/_internal/warp/native" ]; then
    HEADER_COUNT=$(find "$BUNDLE/_internal/warp/native" -name "*.h" | wc -l)
    echo "  OK: warp/native/ ($HEADER_COUNT header files)"
else
    echo "  MISSING: warp/native/ directory (JIT compilation will fail)"
    FAIL=1
fi

if [ -f "$BUNDLE/_internal/config/default_config.yaml" ]; then
    echo "  OK: config/default_config.yaml"
else
    echo "  WARN: config/default_config.yaml not found in bundle"
fi

if [ $FAIL -ne 0 ]; then
    echo ""
    echo "BUILD VERIFICATION FAILED — see above"
    exit 1
fi

echo ""
echo "Bundle size: $(du -sh "$BUNDLE" | cut -f1)"

# ── Create distributable archive ──────────────────────────────
ARCHIVE="$DIST_DIR/${APP_NAME}-${VERSION}-linux-x86_64.tar.gz"
cd "$DIST_DIR"
tar -czf "$ARCHIVE" "$APP_NAME/"
echo "Archive:     $ARCHIVE ($(du -sh "$ARCHIVE" | cut -f1))"
echo ""
echo "=== Build complete ==="
echo ""
echo "To test:  $BUNDLE/$APP_NAME"
