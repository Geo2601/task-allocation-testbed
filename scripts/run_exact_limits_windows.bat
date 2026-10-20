@echo off
rem Limits of the exact solution (article, section 2): up to which size CP-SAT proves optimality.
rem Run by double-click or from cmd. Extra options go to experiments.exact_limits,
rem for example:  run_exact_limits_windows.bat --time-limit 300
rem The run can be stopped (Ctrl+C or close the window) and started again:
rem finished examples are skipped. Results: results\exact_limits\exact_limits_*.csv
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."

call scripts\setup_windows.bat
if errorlevel 1 (
  pause
  exit /b 1
)
echo Run. Long: leave the computer on, sleep mode off.
.venv\Scripts\python.exe -m experiments.exact_limits %*
.venv\Scripts\python.exe -m experiments.exact_limits_summary
echo Done. Results are in results\exact_limits
pause
