@echo off
cd /d "%~dp0"
echo Starting Supply Chain Copilot dashboard...
echo Your browser will open automatically. Close this window to stop.
".venv\Scripts\python.exe" -m streamlit run supplychain_copilot/dashboard/app.py
pause
