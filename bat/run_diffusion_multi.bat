@echo off
setlocal
REM Usage: run_diffusion_multi.bat <model_path> <type> <game>
REM <type> should be "regular"
REM <game> should be "MM", "SMB1", etc. Used to select the correct tileset.
REM This script runs all standard run_diffusion.py calls for a given model and type.

set MODEL_PATH=%1
set TYPE=%2
set GAME=%3

if "%MODEL_PATH%"=="" (
    echo ERROR: Must provide model_path as first argument.
    exit /b 1
)
if "%TYPE%"=="" set TYPE=regular
if "%GAME%"=="" set GAME=MM

REM Select tileset based on game
set TILESET=smb.json
if /I "%GAME%"=="MM" set TILESET=extended_tiles.json

set UNCOND_OUTPUT=%MODEL_PATH%-unconditional-samples

python run_diffusion.py --model_path %MODEL_PATH% --num_samples 100 --save_as_json --output_format image --tileset %TILESET% --output_dir "%UNCOND_OUTPUT%-short"
python run_diffusion.py --model_path %MODEL_PATH% --num_samples 100 --save_as_json --output_format image --tileset %TILESET% --output_dir "%UNCOND_OUTPUT%-long" --level_width 128
