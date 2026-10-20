@echo off
rem Robustness checks (article, section 4). Usage:  run_checks_windows.bat [N]   (N examples per cell, default 50)
rem   clusters_half   - 4 clusters for 8 vehicles (fewer clusters than vehicles)
rem   clusters_double - 16 clusters for 8 vehicles (more clusters than vehicles)
rem   purity          - mixed-type group, every cluster holds tasks of one type (hypothesis H4)
rem 40 cells x N examples, M = 8, 48 tasks, best known reference. About 2 hours on 12 logical cores.
rem The run can be stopped at any moment (close the window) and started again with the same command.
rem Results: results\checks\*.json, table results\checks\all.csv
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
set OUT=results/checks
echo [1/2] Checks: number of clusters and one type per cluster; one process per logical core
.venv\Scripts\python.exe -m experiments.run_main --set clusters_half,clusters_double,purity --reps-to %REPS% --jobs %NUMBER_OF_PROCESSORS% --workers 1 --bk-time 5 --exact-s-max-n 0 --out %OUT%
if errorlevel 1 goto failed
echo [2/2] Table
.venv\Scripts\python.exe -m experiments.run_main --collect --out %OUT%
echo All done. Results are in results\checks (all.csv and *.json).
pause
exit /b 0
:failed
echo Stopped with an error. See the messages above.
pause
exit /b 1
