REM @echo off
REM Usage: run_full_pipeline.bat <input> <model_path> [type] [game] [seed]
REM <input>      path to a .txt ASCII level file or folder of .txt files
REM <model_path> path to the diffusion model
REM [type]       defaults to "regular"
REM [game]       defaults to "MM"
REM [seed]       defaults to 0

set INPUT=%1
set MODEL_PATH=%2
set TYPE=%3
set GAME=%4
set SEED=%5

if "%INPUT%"=="" (
    echo ERROR: Must provide input path as first argument.
    exit /b 1
)
if "%MODEL_PATH%"=="" (
    echo ERROR: Must provide model_path as second argument.
    exit /b 1
)
if "%TYPE%"=="" set TYPE=regular
if "%GAME%"=="" set GAME=MM
if "%SEED%"=="" set SEED=0

echo === Step 1: Preparing dataset with LLM captions ===
call prepare-mario-maker-llm.bat "%INPUT%" %SEED%
if %ERRORLEVEL% neq 0 (
    echo ERROR: prepare-mario-maker-llm.bat failed.
    exit /b 1
)

echo === Step 2: Running diffusion generation ===
call run_diffusion_multi.bat "%MODEL_PATH%" %TYPE% %GAME%
if %ERRORLEVEL% neq 0 (
    echo ERROR: run_diffusion_multi.bat failed.
    exit /b 1
)

echo === Pipeline complete! ===
