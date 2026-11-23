@echo off
cd /d "%~dp0"
start http://127.0.0.1:5000
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m src.web_app
) else (
    python -m src.web_app
)
pause
