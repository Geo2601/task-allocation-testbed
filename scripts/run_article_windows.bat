@echo off
rem Reproduces Tables 1-5 and Figures 1-4 of the article from the data folder and checks
rem every number of the text against the recomputed value. About 5 minutes.
rem Results: results\article (report.md lists every checked number).
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."
call scripts\setup_windows.bat
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m experiments.article --jobs %NUMBER_OF_PROCESSORS%
echo Results are in results\article (see report.md).
pause
exit /b 0
:failed
pause
exit /b 1
