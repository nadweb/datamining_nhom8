@echo off
cd /d C:\Users\LENOVO\Documents\ScrapeData
set PYTHONUTF8=1
call venv\Scripts\activate
python main.py collect >> logs\cron.log 2>&1