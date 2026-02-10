@echo off
REM Phase 3a Training Orchestrator - Windows Batch Launcher
REM Simple interface for launching monitoring tools

setlocal enabledelayedexpansion

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please ensure Python is installed and in PATH.
    pause
    exit /b 1
)

REM Get script directory
set SCRIPT_DIR=%~dp0

REM Display menu
:menu
cls
echo.
echo ================================================================================
echo  🚀 PHASE 3a TRAINING ORCHESTRATOR
echo ================================================================================
echo.
echo Current Directory: %SCRIPT_DIR%
echo.
echo SELECT MONITORING MODE:
echo.
echo  1] Full Orchestration (RECOMMENDED)
echo     - Starts monitor and checker
echo     - Waits up to 20 minutes for completion
echo     - Auto-displays results and alerts
echo.
echo  2] Wait Mode
echo     - Monitor in background
echo     - Blocks until complete
echo     - Simpler alternative to #1
echo.
echo  3] Monitor Only
echo     - Real-time progress tracking
echo     - Shows epoch/loss every 30 seconds
echo.
echo  4] Status Checker - Periodic (Every 30s)
echo     - Lightweight status snapshots
echo     - Runs in background, updates every 30 seconds
echo.
echo  5] Status Checker - Once
echo     - Single status check
echo     - Shows current state and exits
echo.
echo  6] Trigger Alerts (Manual)
echo     - Manually trigger completion alerts
echo     - Use when training already complete
echo.
echo  7] View Monitor Log
echo     - Display latest monitor output
echo.
echo  8] View Orchestrator Log
echo     - Display latest orchestrator output
echo.
echo  Q] Quit
echo.
echo ================================================================================
set /p choice="Enter selection (1-8, Q to quit): "

if /i "!choice!"=="Q" goto end
if /i "!choice!"=="1" goto option1
if /i "!choice!"=="2" goto option2
if /i "!choice!"=="3" goto option3
if /i "!choice!"=="4" goto option4
if /i "!choice!"=="5" goto option5
if /i "!choice!"=="6" goto option6
if /i "!choice!"=="7" goto option7
if /i "!choice!"=="8" goto option8

echo.
echo ❌ Invalid selection. Please try again.
timeout /t 2 /nobreak
goto menu

:option1
echo.
echo 🚀 Starting Full Orchestration...
echo (This will run for up to 20 minutes)
echo.
cd /d "%SCRIPT_DIR%"
python orchestrator_phase3a.py --full
pause
goto menu

:option2
echo.
echo ⏳ Starting Wait Mode...
echo (Monitoring background training for completion)
echo.
cd /d "%SCRIPT_DIR%"
python orchestrator_phase3a.py --wait
pause
goto menu

:option3
echo.
echo 📊 Starting Monitor...
echo (Real-time progress tracking, auto-exits on completion)
echo.
cd /d "%SCRIPT_DIR%"
python monitor_phase3a_training.py
pause
goto menu

:option4
echo.
echo 📈 Starting Status Checker (Periodic)...
echo (Updates every 30 seconds, runs for ~20 minutes)
echo.
cd /d "%SCRIPT_DIR%"
python status_checker_phase3a.py --periodic 30
pause
goto menu

:option5
echo.
echo 📍 Checking Status (Once)...
echo.
cd /d "%SCRIPT_DIR%"
python status_checker_phase3a.py --once --verbose
echo.
pause
goto menu

:option6
echo.
echo 🔔 Triggering Alert Service...
echo.
cd /d "%SCRIPT_DIR%"
python alert_service_phase3a.py --detailed
echo.
pause
goto menu

:option7
echo.
echo 📋 Monitor Log:
echo ================================================================================
if exist "logs_scaled\monitor.log" (
    type logs_scaled\monitor.log
) else (
    echo (no monitor log found yet)
)
echo ================================================================================
echo.
pause
goto menu

:option8
echo.
echo 📋 Orchestrator Log:
echo ================================================================================
if exist "logs_scaled\orchestrator.log" (
    type logs_scaled\orchestrator.log
) else (
    echo (no orchestrator log found yet)
)
echo ================================================================================
echo.
pause
goto menu

:end
echo.
echo Goodbye!
echo.
exit /b 0
