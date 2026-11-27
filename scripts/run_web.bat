@echo off
cd /d "%~dp0.."
title FlightRisk VN - Web App Server
echo ========================================================
echo        FLIGHTRISK VN - HE THONG DU BAO CHUYEN BAY
echo ========================================================
echo Dang khoi dong may chu tai http://127.0.0.1:5000 ...
echo.
start "" http://127.0.0.1:5000
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe main.py web
) else (
    python main.py web
)
pause
