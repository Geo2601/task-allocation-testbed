@echo off
rem Creates the Python environment in .venv, installs the packages and runs a quick self-check.
rem Other scripts call it automatically when .venv is missing. Needs Python 3.11-3.13 from python.org.
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."
if exist .venv\Scripts\python.exe goto check
echo Creating Python environment in .venv ...
py -3.13 -m venv .venv 2>nul || py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || py -3 -m venv .venv 2>nul || python -m venv .venv
if not exist .venv\Scripts\python.exe (
  echo Python 3.11-3.13 not found. Install Python 3.13 from python.org and run again.
  exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo Package installation failed. See the messages above.
  rmdir /s /q .venv
  exit /b 1
)
:check
echo Quick self-check (about a minute) ...
.venv\Scripts\python.exe -m pytest -q -x tests/test_core.py tests/test_exact.py
if errorlevel 1 (
  echo Self-check failed. See the messages above.
  exit /b 1
)
exit /b 0
