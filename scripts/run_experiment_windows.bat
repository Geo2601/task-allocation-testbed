@echo off
rem Main experiment. Usage:  run_experiment_windows.bat [N]   (N examples per cell, default 50)
rem The run can be stopped at any moment (close the window) and started again with the same command:
rem finished examples are skipped. Run it night after night until it prints "All done".
rem Results: results\experiment\*.json, table results\experiment\all.csv
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."
set REPS=%1
if "%REPS%"=="" set REPS=50
call scripts\setup_windows.bat
if errorlevel 1 (
  pause
  exit /b 1
)
set OUT=results/experiment
echo [1/4] Exact references, N up to 24: 2 processes x 6 CP-SAT threads
.venv\Scripts\python.exe -m experiments.run_main --set main --max-n 24 --reps-to %REPS% --jobs 2 --workers 6 --exact-time 60 --out %OUT%
if errorlevel 1 goto failed
echo [2/4] Best known references, N from 48: one process per logical core
.venv\Scripts\python.exe -m experiments.run_main --set all --min-n 25 --reps-to %REPS% --jobs %NUMBER_OF_PROCESSORS% --workers 1 --bk-time 5 --exact-s-max-n 0 --out %OUT%
if errorlevel 1 goto failed
echo [3/4] Proven minimum of total path for N = 48 on examples 0-4
.venv\Scripts\python.exe -m experiments.refine_exact_s --out %OUT% --max-n 48 --reps-to 5 --workers %NUMBER_OF_PROCESSORS%
if errorlevel 1 goto failed
echo [4/4] Table and analysis
.venv\Scripts\python.exe -m experiments.run_main --collect --out %OUT%
.venv\Scripts\python.exe -m experiments.analyze_main --src %OUT%/all.csv > %OUT%/analysis.txt
echo All done. Results are in results\experiment.
pause
exit /b 0
:failed
echo Stopped with an error. See the messages above.
pause
exit /b 1
