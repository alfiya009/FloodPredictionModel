@echo off
echo ===== Starting weather forecast update at %date% %time% =====
echo.

cd /d C:\Users\alfiy\Desktop\model\model

if not exist venv\Scripts\activate.bat (
  echo ERROR: Virtual environment not found. Please create it first.
  exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Running forecast update script...
python 7days.py

if %ERRORLEVEL% NEQ 0 (
  echo ERROR: Forecast update failed with error code %ERRORLEVEL%
  exit /b %ERRORLEVEL%
)

echo.
echo ===== Forecast update completed successfully at %date% %time% =====

REM Create a log entry
echo %date% %time% - Forecast update completed successfully >> update_forecast_log.txt
