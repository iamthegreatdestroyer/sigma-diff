@echo off
REM ============================================================================
REM Ryzanstein MCP Server Auto-Startup Script
REM ============================================================================
REM This script ensures the MCP gRPC backend server starts automatically
REM and runs with proper error handling and logging.
REM
REM Usage: START_MCP_SERVER.bat [--debug]
REM ============================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM Configuration
set MCP_EXECUTABLE=mcp-server.exe
set MCP_LOG_DIR=%USERPROFILE%\AppData\Local\Ryzanstein\logs
set MCP_LOG_FILE=!MCP_LOG_DIR!\mcp-server.log
set MAX_RETRIES=3
set RETRY_DELAY=2000
set DEBUG=%1

REM Create log directory if it doesn't exist
if not exist "!MCP_LOG_DIR!" mkdir "!MCP_LOG_DIR!"

REM Log function
:log_message
if not "!DEBUG!"=="" (
    echo [%DATE% %TIME%] %1
)
echo [%DATE% %TIME%] %1 >> "!MCP_LOG_FILE!"
goto :eof

REM ============================================================================
REM Main Startup Logic
REM ============================================================================

call :log_message "=========================================="
call :log_message "MCP Server Auto-Startup Script Started"
call :log_message "=========================================="

REM Check if MCP server executable exists
if not exist "%MCP_EXECUTABLE%" (
    call :log_message "ERROR: MCP server executable not found: %MCP_EXECUTABLE%"
    call :log_message "Current directory: %CD%"
    exit /b 1
)

call :log_message "MCP executable found: %CD%\%MCP_EXECUTABLE%"

REM Check if MCP is already running
call :check_if_running
if %ERRORLEVEL% equ 0 (
    call :log_message "MCP server is already running (port 50051 listening)"
    exit /b 0
)

REM Try to start MCP server
set RETRY_COUNT=0
:retry_loop
if %RETRY_COUNT% geq %MAX_RETRIES% (
    call :log_message "ERROR: Failed to start MCP server after %MAX_RETRIES% attempts"
    exit /b 1
)

set /a RETRY_COUNT+=1
call :log_message "Attempt %RETRY_COUNT%/%MAX_RETRIES%: Starting MCP server..."

REM Start MCP server in background
start "Ryzanstein MCP Server" /MIN "%MCP_EXECUTABLE%"

REM Wait for server to start
timeout /t 2 /nobreak

REM Check if server is now running
call :check_if_running
if %ERRORLEVEL% equ 0 (
    call :log_message "SUCCESS: MCP server started successfully"
    call :log_message "MCP server listening on port 50051"
    exit /b 0
)

call :log_message "MCP server not responding yet, will retry..."
timeout /t %RETRY_DELAY% /nobreak
goto retry_loop

REM ============================================================================
REM Helper Functions
REM ============================================================================

REM Check if MCP server is running and port 50051 is listening
:check_if_running
netstat -ano | find /I ":50051" | find /I "LISTENING" >nul 2>&1
exit /b %ERRORLEVEL%

REM ============================================================================
REM Error Handling
REM ============================================================================

:error_handler
call :log_message "ERROR: An unexpected error occurred"
call :log_message "Error Code: %ERRORLEVEL%"
exit /b 1
