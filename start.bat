@echo off
setlocal
set "ROOT=C:\StudioVoice"
set "VENV=%ROOT%\voiceenv"
set "PYTHON=%VENV%\Scripts\python.exe"
set "CUDA_VISIBLE_DEVICES=-1"
set "PYTORCH_NO_CUDA_MEMORY_CACHING=1"
set "PHONEMIZER_ESPEAK_LIBRARY=%VENV%\Lib\site-packages\espeakng_loader\espeak-ng.dll"
set "ESPEAK_DATA_PATH=%VENV%\Lib\site-packages\espeakng_loader\espeak-ng-data"

if not exist "%PYTHON%" (
  echo Virtual environment not found at %VENV%.
  pause
  exit /b 1
)

cd /d "%ROOT%"
"%PYTHON%" "%ROOT%\studio_voice.py"
