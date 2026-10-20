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

echo [DubClean] Подготовка локального окружения Python 3.11...

if exist "%ROOT%.venv\Scripts\python.exe" (
  "%ROOT%.venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
  if not errorlevel 1 goto dependencies
)

echo [DubClean] Существующее окружение отсутствует или не запускается; оно будет пересобрано.
if exist "%ROOT%tools\python311\python.exe" goto create_with_portable
py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
if not errorlevel 1 goto create_with_py
python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
if not errorlevel 1 goto create_with_python

echo [DubClean] Python 3.11 не найден.
echo [DubClean] Установите 64-разрядный Python 3.11 и повторите запуск.
goto failed

:create_with_portable
"%ROOT%tools\python311\python.exe" -m venv --clear "%ROOT%.venv"
if errorlevel 1 goto failed
goto verify_environment

:create_with_py
if exist "%ROOT%.venv" (
  py -3.11 -m venv --clear "%ROOT%.venv"
) else (
  py -3.11 -m venv "%ROOT%.venv"
)
if errorlevel 1 goto failed
goto verify_environment

:create_with_python
if exist "%ROOT%.venv" (
  python -m venv --clear "%ROOT%.venv"
) else (
  python -m venv "%ROOT%.venv"
)
if errorlevel 1 goto failed

:verify_environment
if not exist "%ROOT%.venv\Scripts\python.exe" goto failed
"%ROOT%.venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
if errorlevel 1 goto failed

:dependencies
"%ROOT%.venv\Scripts\python.exe" -m ensurepip --upgrade >nul 2>nul
"%ROOT%.venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed

"%ROOT%.venv\Scripts\python.exe" -c "import torch, torchaudio, torchvision; assert torch.__version__.split('+')[0] == '2.5.1'; assert torchaudio.__version__.split('+')[0] == '2.5.1'; assert torchvision.__version__.split('+')[0] == '0.20.1'" >nul 2>nul
if not errorlevel 1 goto base_requirements

echo [DubClean] Установка PyTorch 2.5.1 с поддержкой CUDA 12.1...
"%ROOT%.venv\Scripts\python.exe" -m pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
if not errorlevel 1 goto base_requirements

echo [DubClean] CUDA-сборка недоступна; устанавливается CPU-сборка PyTorch...
"%ROOT%.venv\Scripts\python.exe" -m pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed

:base_requirements
"%ROOT%.venv\Scripts\python.exe" -m pip install -r "%ROOT%requirements.txt"
if errorlevel 1 goto failed

"%ROOT%.venv\Scripts\python.exe" "%ROOT%portable_tools\download_models.py"
if errorlevel 1 goto failed
"%ROOT%.venv\Scripts\python.exe" "%ROOT%portable_tools\install_ffmpeg.py"
if errorlevel 1 goto failed
if exist "%ROOT%tools\ffmpeg\bin\ffmpeg.exe" set "PATH=%ROOT%tools\ffmpeg\bin;%PATH%"

"%ROOT%.venv\Scripts\python.exe" "%ROOT%portable_tools\configure_portable.py"
if errorlevel 1 goto failed
"%ROOT%.venv\Scripts\python.exe" "%ROOT%portable_tools\check_install.py" --quick
if errorlevel 1 goto failed

echo [DubClean] Окружение готово. Запустите START_DUBCLEAN.bat.
if not "%DUBCLEAN_NO_PAUSE%"=="1" pause
exit /b 0

:failed
echo [DubClean] Подготовка окружения не завершена.
if not "%DUBCLEAN_NO_PAUSE%"=="1" pause
exit /b 1
