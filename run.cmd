@echo off
REM Wrapper, damit run.ps1 auch bei restriktiver PowerShell-Execution-Policy laeuft.
REM   run.cmd -Image LGM\data_test\bird_rgba.png
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
