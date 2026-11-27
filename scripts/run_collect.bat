@echo off
cd /d "%~dp0.."
set PYTHONUTF8=1
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe main.py collect >> logs\cron.log 2>&1
) else (
    python main.py collect >> logs\cron.log 2>&1
)
