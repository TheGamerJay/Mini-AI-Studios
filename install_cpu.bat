@echo off
setlocal
title Mini AI Studios — CPU Install

echo.
echo  ╔══════════════════════════════════════╗
echo  ║     Mini AI Studios — CPU Install    ║
echo  ╚══════════════════════════════════════╝
echo.

:: Create virtual environment
echo [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 ( echo ERROR: Python not found. Install Python 3.10+ first. & pause & exit /b 1 )

:: Activate
call venv\Scripts\activate.bat

:: PyTorch CPU build
echo.
echo [2/3] Installing PyTorch (CPU only)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

:: Other deps
echo.
echo [3/3] Installing dependencies...
pip install -r requirements.txt

echo.
echo  ╔══════════════════════════════════════╗
echo  ║  Done!  Run:  python app.py          ║
echo  ║  Then open:   http://localhost:7860  ║
echo  ╚══════════════════════════════════════╝
echo.
echo  NOTE: First song generation downloads ~1.5 GB of models.
echo  Models are cached — subsequent runs start immediately.
echo.
pause
