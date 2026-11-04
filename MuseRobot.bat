@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Muse 2 Robot Control
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
echo     Muse 2  -^>  RosmasterPi A1 Robot Control
echo  ==============================================
echo     Start BlueMuse and press "Start Streaming" first.
echo     Current profile: %NAME%
echo.
echo     1. Check signal   ^(NaN, contact quality, 20 s^)
echo     2. Calibrate      ^(about 3 minutes^)
echo     3. Run            - show commands on screen only
echo     4. Run            - send to a SERIAL device ^(COM port^)
echo     5. Run            - drive the YAHBOOM ROBOT over Wi-Fi
echo     6. Change profile name
echo     7. Test without headset ^(simulator^)
echo     8. Test robot connection ^(keyboard, no headset^)
echo     0. Exit
echo.
echo     While running: press Ctrl+C, then N, to come back here.
echo.
set "CH="
set /p "CH=  Choose 0-8: "
if "%CH%"=="1" goto :check
if "%CH%"=="2" goto :calibrate
if "%CH%"=="3" goto :run_print
if "%CH%"=="4" goto :run_serial
if "%CH%"=="5" goto :run_robot
if "%CH%"=="6" goto :profile
if "%CH%"=="7" goto :simtest
if "%CH%"=="8" goto :robot_test
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

:run_serial
call :need_profile
set "PORT=COM5"
if exist "profiles\last_port.txt" set /p PORT=<"profiles\last_port.txt"
set /p "PORT=  COM port [%PORT%]: "
set "BAUD=115200"
set /p "BAUD=  Baud rate [%BAUD%]: "
> "profiles\last_port.txt" echo %PORT%
"%VPY%" -m muse_robot run --profile "profiles\%NAME%.json" --output print --output serial:%PORT%:%BAUD% --sound
goto :done

:run_robot
call :need_profile
call :ask_robot
echo  Start robot/start_bridge.sh on the robot first. Jaw clench = STOP.
"%VPY%" -m muse_robot run --profile "profiles\%NAME%.json" --output print --output robot:%ROBOT_IP%:9000 --sound
goto :done

:robot_test
call :ask_robot
"%VPY%" -m muse_robot robot-test %ROBOT_IP%
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
rem  helpers (called with "call :name", they return with exit /b)
rem ============================================================
:ask_profile
set /p "NAME=  Profile name, e.g. your nickname [%NAME%]: "
set "NAME=%NAME: =_%"
> "profiles\last.txt" echo %NAME%
exit /b 0

:ask_robot
set "ROBOT_IP=192.168.1.11"
if exist "profiles\last_robot_ip.txt" set /p ROBOT_IP=<"profiles\last_robot_ip.txt"
set /p "ROBOT_IP=  Robot IP address [%ROBOT_IP%]: "
> "profiles\last_robot_ip.txt" echo %ROBOT_IP%
exit /b 0

:need_profile
if exist "profiles\%NAME%.json" exit /b 0
echo.
echo  Profile "%NAME%" is not calibrated yet - rough default thresholds will be used.
echo  Choose 2 ^(Calibrate^) first for good accuracy.
echo.
pause
exit /b 0
