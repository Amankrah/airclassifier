@echo off
REM ProteinProcessIO — Windows build script
setlocal

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set DIST_DIR=%PROJECT_ROOT%\dist
set BUILD_DIR=%PROJECT_ROOT%\build
set VENV=%PROJECT_ROOT%\venv
set APP_NAME=ProteinProcessIO
set VERSION=1.0.0

echo === %APP_NAME% Windows Build ===
echo Project root: %PROJECT_ROOT%
echo.

REM Activate venv
call "%VENV%\Scripts\activate.bat"

REM Ensure PyInstaller is installed
pip install --quiet "pyinstaller>=6.0"

REM Clean previous build
rd /s /q "%BUILD_DIR%\%APP_NAME%" 2>nul
rd /s /q "%DIST_DIR%\%APP_NAME%" 2>nul

echo Running PyInstaller...
pyinstaller "%SCRIPT_DIR%%APP_NAME%.spec" ^
    --distpath "%DIST_DIR%" ^
    --workpath "%BUILD_DIR%" ^
    --noconfirm

echo.
echo === Verifying build ===
set BUNDLE=%DIST_DIR%\%APP_NAME%

if exist "%BUNDLE%\%APP_NAME%.exe" (
    echo   OK: %APP_NAME%.exe
) else (
    echo   MISSING: %APP_NAME%.exe
    exit /b 1
)

if exist "%BUNDLE%\_internal\warp\bin\warp.dll" (
    echo   OK: warp.dll
) else (
    echo   WARN: warp.dll not found (check warp/bin/ contents)
)

if exist "%BUNDLE%\_internal\warp\bin\warp-clang.dll" (
    echo   OK: warp-clang.dll
) else (
    echo   WARN: warp-clang.dll not found
)

echo.
echo === Build complete ===
echo.
echo To test:  "%BUNDLE%\%APP_NAME%.exe"

endlocal
