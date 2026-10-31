@echo off
title Integrated Machine Learning Dashboard
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
echo ==========================================
echo   Integrated Machine Learning Dashboard
echo ==========================================
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate
echo Installing required packages...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Starting dashboard...
python app.py
pause
