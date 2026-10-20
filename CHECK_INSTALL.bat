@echo off
setlocal
chcp 65001 >nul
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%ROOT%python_src"
if exist "%ROOT%tools\ffmpeg\bin\ffmpeg.exe" (
  set "PATH=%ROOT%tools\ffmpeg\bin;%PATH%"
)

if not exist "%ROOT%.venv\Scripts\python.exe" (
  echo [DubClean] Локальное окружение не найдено. Сначала запустите INSTALL_DEPS.bat.
  if not "%DUBCLEAN_NO_PAUSE%"=="1" pause
  exit /b 1
)

"%ROOT%.venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
if errorlevel 1 (
  echo [DubClean] Окружение повреждено или создано другой версией Python.
  echo [DubClean] Запустите INSTALL_DEPS.bat для восстановления.
  if not "%DUBCLEAN_NO_PAUSE%"=="1" pause
  exit /b 1
)

"%ROOT%.venv\Scripts\python.exe" "%ROOT%portable_tools\configure_portable.py"
if errorlevel 1 (
  if not "%DUBCLEAN_NO_PAUSE%"=="1" pause
  exit /b 1
)

"%ROOT%.venv\Scripts\python.exe" "%ROOT%portable_tools\check_install.py"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%DUBCLEAN_NO_PAUSE%"=="1" pause
exit /b %EXIT_CODE%
