@echo off
setlocal enabledelayedexpansion

:: ============================================================
::  Experiment runner — 10 seeded runs, 4 parallel islands
::  Usage:  run_experiment.bat <instance_index>
::  Example: run_experiment.bat 0    (bonus1000)
::           run_experiment.bat 1    (bubbles1)
:: ============================================================

set EXE=C:\Users\Demir\researchproject\MA-CETSP\build\Release\MA-CETSP.exe
set OUTDIR=C:\Users\Demir\researchproject\MA-CETSP\solutions\experiment_results
set ISLANDS=4
set ITERATIONS=200

:: Instance index from first argument, default 0
set INSTANCE=%~1
if "%INSTANCE%"=="" set INSTANCE=0

:: Seeds: 1, 11, 21, 31, 41, 51, 61, 71, 81, 91
set SEEDS=1 11 21 31 41 51 61 71 81 91

mkdir "%OUTDIR%" 2>nul

set LOGFILE=%OUTDIR%\instance%INSTANCE%_results.txt
echo Experiment: instance=%INSTANCE%  islands=%ISLANDS%  iterations=%ITERATIONS% > "%LOGFILE%"
echo Started: %date% %time% >> "%LOGFILE%"
echo. >> "%LOGFILE%"
echo Run   Seed   GlobalBest   BestModel   WallTime(s) >> "%LOGFILE%"
echo -------------------------------------------------- >> "%LOGFILE%"

echo.
echo ====================================================
echo  Running 10 seeds on instance %INSTANCE% (%ISLANDS% islands each)
echo ====================================================
echo.

set RUN=0
for %%S in (%SEEDS%) do (
    set /a RUN+=1
    echo [Run !RUN!/10] seed=%%S ...

    set RUNLOG=%OUTDIR%\instance%INSTANCE%_run!RUN!_seed%%S.log

    "%EXE%" -i %INSTANCE% -s %%S --islands %ISLANDS% -r %ITERATIONS% > "!RUNLOG!" 2>&1

    :: Extract GLOBAL BEST value from summary line
    set BEST=N/A
    set MODEL=N/A
    set WTIME=N/A
    for /f "tokens=*" %%L in ('findstr /C:"GLOBAL BEST" "!RUNLOG!"') do (
        set LINE=%%L
    )
    :: Parse: "  GLOBAL BEST: island=X  model=Y  value=Z  time=Ws  rejected=N"
    for /f "tokens=1,2,3,4,5,6,7,8,9,10 delims= " %%a in ("!LINE!") do (
        set T1=%%c & set T2=%%d & set T3=%%e & set T4=%%f & set T5=%%g
    )
    for /f "tokens=2 delims==" %%V in ("!T3!") do set BEST=%%V
    for /f "tokens=2 delims==" %%V in ("!T2!") do set MODEL=%%V
    for /f "tokens=2 delims==" %%V in ("!T4!") do set WTIME=%%V

    echo !RUN!     %%S     !BEST!     !MODEL!     !WTIME! >> "%LOGFILE%"
    echo   -> best=!BEST!  model=!MODEL!  time=!WTIME!
)

echo.
echo ====================================================
echo  All runs complete. Results saved to:
echo  %LOGFILE%
echo ====================================================
echo.
type "%LOGFILE%"

endlocal
