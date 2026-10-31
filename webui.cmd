@echo off
REM Startet die LGM-Oberflaeche und oeffnet http://127.0.0.1:7860
cd /d "%~dp0"
".venv\Scripts\python.exe" webui.py %*
