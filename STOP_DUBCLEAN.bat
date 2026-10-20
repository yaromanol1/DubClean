@echo off
setlocal
chcp 65001 >nul
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%ROOT%python_src"

if not exist "%ROOT%.venv\Scripts\python.exe" (
  echo [DubClean] Локальное окружение не найдено; останавливать нечего.
  if not "%DUBCLEAN_NO_PAUSE%"=="1" pause
  exit /b 0
)

"%ROOT%.venv\Scripts\python.exe" "%ROOT%portable_tools\configure_portable.py" >nul
if errorlevel 1 (
  echo [DubClean] Не удалось прочитать конфигурацию. Запустите CHECK_INSTALL.bat.
  if not "%DUBCLEAN_NO_PAUSE%"=="1" pause
  exit /b 1
)

echo [DubClean] Остановка сервиса и активных процессов...
"%ROOT%.venv\Scripts\python.exe" -m experiments.paired_reference_cancel.service_control
set "EXIT_CODE=%ERRORLEVEL%"
if not "%DUBCLEAN_NO_PAUSE%"=="1" pause
exit /b %EXIT_CODE%
