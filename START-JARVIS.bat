@echo off
setlocal

echo.
echo ==============================================
echo        J.A.RVIS System
echo ==============================================
echo.

set "BACKEND=%~dp0backend"

cd /d "%~dp0"
echo Backend: %BACKEND%
echo.

REM Check if venv exists
if exist "%BACKEND%\.venv\Scripts\python.exe" (
    echo Using Python from .venv
    set "PYTHON=%BACKEND%\.venv\Scripts\python.exe"
) else (
    echo Using system Python from PATH
    set "PYTHON=python"
)

echo Using: %PYTHON%
echo.

REM Launch JARVIS with embedded launcher (tray + greeting)
echo Starting JARVIS...
echo.
cd /d "%BACKEND%"
"%PYTHON%" jarvis_embedded.py

echo.
echo JARVIS has stopped.
echo.
pause