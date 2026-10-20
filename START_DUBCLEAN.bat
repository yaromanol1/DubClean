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

if exist "%ROOT%.venv\Scripts\python.exe" (
  "%ROOT%.venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
)
if not exist "%ROOT%.venv\Scripts\python.exe" goto install
if errorlevel 1 goto install
goto configure

:install
echo [DubClean] Подготовка локального окружения Python...
call "%ROOT%SETUP.bat"
if errorlevel 1 goto failed

:configure
"%ROOT%.venv\Scripts\python.exe" "%ROOT%portable_tools\service_status.py"
if errorlevel 1 goto configure_paths
exit /b 0

:configure_paths
rem Update portable paths before importing the backend after relocating the folder.
"%ROOT%.venv\Scripts\python.exe" "%ROOT%portable_tools\configure_portable.py"
if errorlevel 1 goto failed

"%ROOT%.venv\Scripts\python.exe" -c "import torch, torchaudio, numpy, scipy, soundfile, flask, yaml, clearvoice" >nul 2>nul
if errorlevel 1 (
  echo [DubClean] Python-зависимости неполны. Запускается восстановление...
  call "%ROOT%INSTALL_DEPS.bat"
  if errorlevel 1 goto failed
  "%ROOT%.venv\Scripts\python.exe" "%ROOT%portable_tools\configure_portable.py"
  if errorlevel 1 goto failed
)

"%ROOT%.venv\Scripts\python.exe" "%ROOT%portable_tools\download_models.py"
if errorlevel 1 goto failed
"%ROOT%.venv\Scripts\python.exe" "%ROOT%portable_tools\install_ffmpeg.py"
if errorlevel 1 goto failed
if exist "%ROOT%tools\ffmpeg\bin\ffmpeg.exe" set "PATH=%ROOT%tools\ffmpeg\bin;%PATH%"
"%ROOT%.venv\Scripts\python.exe" "%ROOT%portable_tools\check_install.py" --quick
if errorlevel 1 (
  echo [DubClean] Запуск отменён. Исправьте ошибки выше или выполните CHECK_INSTALL.bat.
  goto failed
)

echo [DubClean] Сервис запускается: http://127.0.0.1:8768/product
"%ROOT%.venv\Scripts\python.exe" "%ROOT%python_src\experiments\paired_reference_cancel\app.py"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" echo [DubClean] Сервис завершился с кодом %EXIT_CODE%.
if not "%DUBCLEAN_NO_PAUSE%"=="1" pause
exit /b %EXIT_CODE%

:failed
if not "%DUBCLEAN_NO_PAUSE%"=="1" pause
exit /b 1
