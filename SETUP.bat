@echo off
setlocal
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0portable_tools\bootstrap.ps1"
set "RESULT=%ERRORLEVEL%"
if not "%DUBCLEAN_NO_PAUSE%"=="1" pause
exit /b %RESULT%
