@echo off
rem Double-click to open hicscaf Studio in your browser.
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)
echo Starting hicscaf Studio ... (close this window to stop)
"%PY%" -m hicscaf.cli studio %*
if errorlevel 1 (
  echo.
  echo hicscaf Studio stopped with an error. See the message above.
)
pause
