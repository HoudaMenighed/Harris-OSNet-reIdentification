@echo off

cd /d "deep-person-reid"

call "D:\downloads\anaconda\Scripts\activate.bat" torchreid

python setup_datamanager.py

if %errorlevel% equ 0 (
    python setup_model.py
)

exit /b
