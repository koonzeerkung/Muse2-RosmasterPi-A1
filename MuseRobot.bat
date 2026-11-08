@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Muse 2 EEG - robot packets
cd /d "%~dp0"

rem ============================================================
rem  1) Find Python and install packages (first run only)
rem ============================================================
set "VPY=%~dp0.venv\Scripts\python.exe"
if exist "%VPY%" goto :deps

set "PY="
for %%P in (py python python3) do (
    if not defined PY (
        %%P -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1 && set "PY=%%P"
    )
)
for %%D in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3") do (
    if not defined PY if exist "%%~D\python.exe" set "PY=%%~D\python.exe"
)
if not defined PY (
    echo.
    echo  Python 3.9+ was not found.
    echo  Install it from https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^)
    echo  then double-click this file again.
    echo.
    pause
    exit /b 1
)

echo  First run: creating a private Python environment in .venv ...
"%PY%" -m venv .venv
if errorlevel 1 (
    echo  Could not create .venv - using "%PY%" directly instead.
    set "VPY=%PY%"
)

:deps
if exist ".venv\installed.flag" goto :start
echo  Installing packages ^(numpy, scipy, pylsl, pyserial^) - only once ...
"%VPY%" -m pip install --upgrade pip >nul 2>&1
"%VPY%" -m pip install -r requirements.txt pyserial
if errorlevel 1 (
    echo.
    echo  Package install failed. Check your internet connection and try again.
    pause
    exit /b 1
)
if exist ".venv\Scripts" echo ok> ".venv\installed.flag"

:start
if not exist profiles mkdir profiles
set "NAME=default"
if exist "profiles\last.txt" set /p NAME=<"profiles\last.txt"

rem ============================================================
rem  2) Menu
rem ============================================================
:menu
cls
echo.
echo  ==============================================
echo     Muse 2  -^>  robot packets
echo  ==============================================
echo     Start BlueMuse and press "Start Streaming" first.
echo     Current profile: %NAME%
echo.
echo     Jaw clench = TOGGLE_RUN   Double blink = TOGGLE_MODE   Eyebrow raise = TOGGLE_MUSIC
echo.
echo     1. Check signal   ^(NaN, contact quality, 10 s^)
echo     2. Calibrate      ^(about 2 minutes^)
echo     3. Run            - show commands on screen only
echo     4. Run            - send packets to the robot side ^(UDP^)
echo     5. Change profile name
echo     6. Test without headset ^(simulator^)
echo     7. Send test packets ^(keyboard, no headset^)
echo     8. Listen for packets ^(test receiver^)
echo     0. Exit
echo.
echo     While running: press Ctrl+C, then N, to come back here.
echo.
set "CH="
set /p "CH=  Choose 0-8: "
if "%CH%"=="1" goto :check
if "%CH%"=="2" goto :calibrate
if "%CH%"=="3" goto :run_print
if "%CH%"=="4" goto :run_udp
if "%CH%"=="5" goto :profile
if "%CH%"=="6" goto :simtest
if "%CH%"=="7" goto :send_test
if "%CH%"=="8" goto :listen
if "%CH%"=="0" exit /b 0
goto :menu

:check
"%VPY%" -m muse_robot check
goto :done

:calibrate
call :ask_profile
set "TRIALS=8"
set /p "TRIALS=  Trials per gesture [8] (fewer = faster, 5 is the minimum): "
"%VPY%" -m muse_robot calibrate --profile "profiles\%NAME%.json" --trials %TRIALS%
goto :done

:run_print
call :need_profile
"%VPY%" -m muse_robot run --profile "profiles\%NAME%.json" --output print --sound
goto :done

:run_udp
call :need_profile
call :ask_target
"%VPY%" -m muse_robot run --profile "profiles\%NAME%.json" --output udp:%TARGET_IP%:%TARGET_PORT% --sound
goto :done

:send_test
call :ask_target
"%VPY%" -m muse_robot send-test --output udp:%TARGET_IP%:%TARGET_PORT%
goto :done

:listen
set "LPORT=9000"
set /p "LPORT=  UDP port to listen on [9000]: "
"%VPY%" -m muse_robot listen --port %LPORT%
goto :done

:profile
call :ask_profile
goto :menu

:simtest
"%VPY%" -m muse_robot sim-test --profile "profiles\sim.json" --recordings "recordings\sim"
goto :done

:done
echo.
pause
goto :menu

rem ============================================================
rem  helpers - each is used with CALL and returns with exit /b
rem ============================================================
:ask_profile
set /p "NAME=  Profile name, e.g. your nickname [%NAME%]: "
set "NAME=%NAME: =_%"
> "profiles\last.txt" echo %NAME%
exit /b 0

:ask_target
set "TARGET_IP=127.0.0.1"
if exist "profiles\last_target_ip.txt" set /p TARGET_IP=<"profiles\last_target_ip.txt"
set /p "TARGET_IP=  IP address of the robot-side computer [%TARGET_IP%]: "
> "profiles\last_target_ip.txt" echo %TARGET_IP%
set "TARGET_PORT=9000"
set /p "TARGET_PORT=  UDP port [9000]: "
exit /b 0

:need_profile
if exist "profiles\%NAME%.json" exit /b 0
echo.
echo  Profile "%NAME%" is not calibrated yet - rough default thresholds will be used.
echo  Choose 2 ^(Calibrate^) first for good accuracy.
echo.
pause
exit /b 0
