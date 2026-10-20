@echo off
rem Pilot of the main experiment (all sets of cells, 3 examples per cell by default): shows how many
rem examples per cell are needed and how long the full run takes. Extra options go to both phases, for example:
rem   run_pilot_windows.bat --reps-to 5
rem The run can be stopped and started again: finished examples are skipped.
rem Results: results\main\*.json, table results\main\all.csv, summary printed at the end.
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
echo [1/3] Exact fronts, N up to 24: 2 processes x 6 CP-SAT threads
.venv\Scripts\python.exe -m experiments.run_main --set main --max-n 24 --reps-to 3 --jobs 2 --workers 6 --exact-time 60 %*
echo [2/3] Best known fronts, N from 48: one process per logical core
.venv\Scripts\python.exe -m experiments.run_main --set all --min-n 25 --reps-to 3 --jobs %NUMBER_OF_PROCESSORS% --workers 1 --bk-time 5 --exact-time 60 %*
echo [3/3] Summary
.venv\Scripts\python.exe -m experiments.pilot_summary --src results/main/all.csv --cores %NUMBER_OF_PROCESSORS%
echo Done. Table: results\main\all.csv
pause
