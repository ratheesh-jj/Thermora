@echo off
echo Starting THERMORA Live Server...
start .venv\Scripts\python.exe src\thermora_live_server.py

echo Starting Dashboard...
timeout /t 3
start http://localhost:8000
echo If the dashboard didn't open automatically, double click dashboard\index.html in your browser.
pause
