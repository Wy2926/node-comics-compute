@echo off
setlocal
rem Parse the whole block before the updater replaces this file.
(
  if exist "%~dp0.setup\update\transaction.json" (
    "%~dp0.setup\update-host.exe" _update
  ) else (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0source\compute-node\update.ps1" -Root "%~dp0."
  )
  if errorlevel 1 (
    echo Update failed. Read the message above; double-click Update again to retry.
    pause
    exit /b 1
  )
  pause
  exit /b 0
)
