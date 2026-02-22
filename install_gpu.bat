@echo off
setlocal
title Mini AI Studios — GPU Install (CUDA 12.x)

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║   Mini AI Studios — GPU Install (CUDA 12.x)    ║
echo  ╚══════════════════════════════════════════════════╝
echo  Requires: NVIDIA GPU + CUDA 12.x drivers installed
echo.

:: Create virtual environment
echo [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 ( echo ERROR: Python not found. Install Python 3.10+ first. & pause & exit /b 1 )

:: Activate
call venv\Scripts\activate.bat

:: PyTorch CUDA build
echo.
echo [2/3] Installing PyTorch (CUDA 12.1)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

:: Other deps
echo.
echo [3/3] Installing dependencies...
pip install -r requirements.txt

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║  Done!                                          ║
echo  ║  Edit config.py:  DEVICE = "cuda"              ║
echo  ║  Then run:        python app.py                ║
echo  ║  Open:            http://localhost:7860         ║
echo  ╚══════════════════════════════════════════════════╝
echo.
echo  NOTE: First run downloads ~1.5 GB of models (cached after that).
echo  GPU generation is roughly 10x faster than CPU.
echo.
pause
