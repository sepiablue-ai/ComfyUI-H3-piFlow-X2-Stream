@echo off
setlocal
set "COMFY_ROOT=%~1"
if defined COMFY_ROOT if "%COMFY_ROOT:~0,2%"=="--" set "COMFY_ROOT="
set "COMFY_PYTHON=%~2"
if defined COMFY_PYTHON if "%COMFY_PYTHON:~0,2%"=="--" set "COMFY_PYTHON="
if not defined COMFY_ROOT if exist "%~dp0..\..\main.py" for %%I in ("%~dp0..\..") do set "COMFY_ROOT=%%~fI"
if not defined COMFY_ROOT set /p "COMFY_ROOT=ComfyUI folder containing main.py: "
if not exist "%COMFY_ROOT%\main.py" (
  echo ERROR: ComfyUI main.py was not found.
  exit /b 1
)
if defined COMFY_PYTHON goto run
set "PYTHON_COUNT=0"
call :candidate "%COMFY_ROOT%\..\python_embeded\python.exe"
call :candidate "%COMFY_ROOT%\.venv\Scripts\python.exe"
call :candidate "%COMFY_ROOT%\..\.venv\Scripts\python.exe"
call :candidate "%COMFY_ROOT%\venv\Scripts\python.exe"
if not "%PYTHON_COUNT%"=="1" (
  echo ERROR: Specify the ComfyUI Python explicitly as the second argument.
  echo Usage: convert_fused.bat "COMFY_ROOT" "COMFY_PYTHON" [--download-bf16 DIR --delete-bf16] [--bf16 PATH]
  exit /b 1
)
:run
if not exist "%COMFY_PYTHON%" (
  echo ERROR: The specified Python does not exist.
  exit /b 1
)
rem Stop ComfyUI first: the conversion uses the GPU and writes a 21 GB model (plus 40.2 GB while the BF16 model exists).
set "H3_PIFLOW_SETUP_ROOT=%COMFY_ROOT%"
"%COMFY_PYTHON%" -X utf8 "%~dp0tools\merge_pih3_adapter_bf16_int8_convrot.py" %*
exit /b %errorlevel%
:candidate
if not exist "%~1" exit /b 0
set /a PYTHON_COUNT+=1 >nul
set "COMFY_PYTHON=%~1"
exit /b 0
