REM @echo off
REM Usage: run_diffusion_multi.bat <model_path> <type> <game> <text>
REM <type> should be "regular"
REM This script runs all standard run_diffusion.py calls for a given model and type.

set MODEL_PATH=%1
set TYPE=%2
set GAME=%3
set TEXT=%4

if "%MODEL_PATH%"=="" (
    echo ERROR: Must provide model_path as first argument.
    exit /b 1
)
if "%TYPE%"=="" set TYPE=regular

set TEXT_FLAG=
if /I "%TEXT%"=="text" set TEXT_FLAG=--text_conditional

set UNCOND_OUTPUT=%MODEL_PATH%-unconditional-samples

python run_diffusion.py --model_path %MODEL_PATH% --num_samples 100 %TEXT_FLAG% --save_as_json --output_dir "%UNCOND_OUTPUT%-short"
python run_diffusion.py --model_path %MODEL_PATH% --num_samples 100 %TEXT_FLAG% --save_as_json --output_dir "%UNCOND_OUTPUT%-long" --level_width 128
