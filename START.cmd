@echo off
cd /d "%~dp0"
where py >nul 2>&1
if errorlevel 1 (
  python client.py
) else (
  py -3 client.py
)
pause
