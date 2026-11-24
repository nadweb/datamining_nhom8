@echo off
cd /d "%~dp0"
title FlightRisk VN - Web App Server
echo ========================================================
echo        FLIGHTRISK VN - HE THONG DU BAO CHUYEN BAY
echo ========================================================
echo Dang khoi dong may chu tai http://127.0.0.1:5000 ...
echo (De dung web server, chi can dong cua so nay)
echo.

:: Tu dong mo trinh duyet
start "" http://127.0.0.1:5000

:: Chay web server
python -m web.app 5000

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [LOI] Khong the khoi dong bang python mac dinh, dang thu bang venv...
    if exist "venv\Scripts\python.exe" (
        venv\Scripts\python.exe -m web.app 5000
    )
)

pause
