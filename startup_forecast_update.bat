@echo off
echo Waiting for 60 seconds before starting forecast update...
timeout /t 60
echo Starting forecast update (from startup)
start "" "C:\Users\alfiy\Desktop\model\model\update_forecast.bat"
