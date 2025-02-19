@echo off

cd /d "deep-person-reid"

call "D:\downloads\anaconda\Scripts\activate.bat" torchreid

python setup_datamanager7.py

if %errorlevel% equ 0 (
    python setup_model7.py
)

exit /b
