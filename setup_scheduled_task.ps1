# Create a scheduled task to run the update_forecast.bat file daily

# Define the task settings
$taskName = "FloodModel_ForecastUpdate"
$taskDescription = "Updates the 7-day weather forecast data for the Flood Prediction Model"
$batchFilePath = "C:\Users\alfiy\Desktop\model\model\update_forecast.bat"

# Check if the batch file exists
if (-not (Test-Path $batchFilePath)) {
    Write-Error "Batch file not found at: $batchFilePath"
    exit 1
}

# Create a new scheduled task action that runs the batch file
$action = New-ScheduledTaskAction -Execute $batchFilePath

# Set up a trigger to run the task daily at 6:00 AM
$trigger = New-ScheduledTaskTrigger -Daily -At 6am

# Configure settings (run with highest privileges, allow on-demand start)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Register the task (will prompt for credentials)
Write-Host "Creating scheduled task: $taskName"
Write-Host "You will be prompted to enter your Windows username and password..."

# Get current user for principal
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Password -RunLevel Highest

# Register the task
Register-ScheduledTask -TaskName $taskName -Description $taskDescription `
    -Action $action -Trigger $trigger -Settings $settings -Principal $principal

Write-Host "Task created successfully!"
Write-Host "The forecast will be updated daily at 6:00 AM."
Write-Host "You can modify this schedule in the Windows Task Scheduler."
