@echo off

cd /d "deep-person-reid"


python setup_datamanager2.py

if %errorlevel% equ 0 (
    python setup_model2.py
)

exit /b
