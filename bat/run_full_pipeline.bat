@echo off
REM Usage: run_full_pipeline.bat <input> [type] [game] [seed]
REM <input>  path to a .txt ASCII level file or folder of .txt files
REM [type]   defaults to "regular"
REM [game]   defaults to "MM"
REM [seed]   defaults to 0
cd ..

set INPUT=%1
set MODEL=%2
set TYPE=%3
set GAME=%4
set SEED=%5

if "%INPUT%"=="" (
    echo ERROR: Must provide input path as first argument.
    exit /b 1
)
if "%MODEL%"=="" set MODEL=qwen2.5:14b
if "%TYPE%"=="" set TYPE=regular
if "%GAME%"=="" set GAME=MM
if "%SEED%"=="" set SEED=0

set TILESET=smb.json
set NUM_TILES=13
set EVAL_TILESET=mm2_tileset_full.json
set CAPTION_ARGS=
if /I "%GAME%"=="MM" (
    set TILESET=extended_tiles.json
    set NUM_TILES=17
    set EVAL_TILESET=extended_tiles.json
    set CAPTION_ARGS=--grid-format tokens --tileset-we mm2_tileset_we.json
)

set RAW_OUTPUT=datasets\%GAME%_Levels-%TYPE%.json
set CAPTIONED_OUTPUT=datasets\%GAME%_LevelsAndCaptions-%TYPE%.json

REM -- Folder for the grid .txt files actually sent to the LLM, placed next
REM    to the input ASCII folder/file.
for %%I in ("%INPUT%") do set "INPUT_DIR=%%~dpI"
set LLM_ASCII_DIR=%INPUT_DIR%ascii_tokens

set MLM_OUTPUT=%GAME%-MLM-%TYPE%%SEED%
set DIFF_OUTPUT=%GAME%-conditional-%TYPE%%SEED%

REM Used to auto-answer "y" to train_diffusion.py's resume-from-checkpoint prompt.
REM A redirected file (rather than a pipe) keeps %ERRORLEVEL% checks working afterward.
set YES_FILE=%TEMP%\mariover_yes.txt
echo y> "%YES_FILE%"

REM -- Ollama setup ---------------------------------------------------------
ollama list >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Ollama not running. Starting ollama server...
    start /min "" ollama serve
    echo Waiting for Ollama to initialise...
    timeout /t 6 /nobreak >nul
)

echo Pulling %MODEL% ^(no-op if already present^)...
ollama pull %MODEL%
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to pull %MODEL%. Is Ollama running?
    exit /b 1
)

echo === Step 1: Preparing dataset with LLM captions ===
python build_dataset_with_ascii.py --input_file %INPUT% --output %RAW_OUTPUT% --tileset %TILESET% --sliding_window --stride 20 
if %ERRORLEVEL% neq 0 ( echo ERROR: build_dataset_with_ascii.py failed. & exit /b 1 )
python MarioMaker_llm_captions.py --dataset %RAW_OUTPUT% --tileset %TILESET% --output %CAPTIONED_OUTPUT% --model %MODEL% --ascii-output-dir "%LLM_ASCII_DIR%"
if %ERRORLEVEL% neq 0 ( echo ERROR: MarioMaker_llm_captions.py failed. & exit /b 1 )
python split_mario_maker_data.py --json %CAPTIONED_OUTPUT% --seed %SEED%
if %ERRORLEVEL% neq 0 ( echo ERROR: split_mario_maker_data.py failed. & exit /b 1 )
python tokenizer.py save --json_file datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --pkl_file datasets\%GAME%_Tokenizer-%TYPE%.pkl
if %ERRORLEVEL% neq 0 ( echo ERROR: tokenizer.py failed. & exit /b 1 )
python create_mario_maker_random_captions.py --json %CAPTIONED_OUTPUT% --output datasets\%GAME%_RandomTest-%TYPE%.json
if %ERRORLEVEL% neq 0 ( echo ERROR: create_mario_maker_random_captions.py failed. & exit /b 1 )

echo === Step 2: Training MLM model ===
python train_mlm.py --epochs 300 --save_checkpoints --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --test_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-test.json --pkl datasets\%GAME%_Tokenizer-%TYPE%.pkl --output_dir %MLM_OUTPUT% --seed %SEED%
if %ERRORLEVEL% neq 0 (
    echo ERROR: train_mlm.py failed.
    exit /b 1
)

echo === Step 3: Training diffusion model ===
python train_diffusion.py --game %GAME% --save_image_epochs 1000 --augment --text_conditional --output_dir "%DIFF_OUTPUT%" --num_epochs 500 --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --pkl datasets\%GAME%_Tokenizer-%TYPE%.pkl --mlm_model_dir %MLM_OUTPUT% --seed %SEED% < "%YES_FILE%"
if %ERRORLEVEL% neq 0 (
    echo ERROR: train_diffusion.py failed.
    exit /b 1
)

echo === Step 4: Running diffusion generation ===
call bat\run_diffusion_multi.bat "%DIFF_OUTPUT%" %TYPE% %GAME% 
if %ERRORLEVEL% neq 0 (
    echo ERROR: run_diffusion_multi.bat failed.
    exit /b 1
)

echo === Step 5: Evaluating caption adherence ===
call bat\evaluate_caption_adherence_multi.bat "%DIFF_OUTPUT%" %TYPE% %GAME%
if %ERRORLEVEL% neq 0 (
    echo ERROR: evaluate_caption_adherence_multi.bat failed.
    exit /b 1
)

echo === Step 6: Evaluating tile distribution ===
python evaluate_tile_distribution.py --model_path "%DIFF_OUTPUT%" --num_tiles %NUM_TILES% --tileset %EVAL_TILESET% --captions_json datasets\%GAME%_LevelsAndCaptions-%TYPE%.json --seed %SEED%
if %ERRORLEVEL% neq 0 (
    echo ERROR: evaluate_tile_distribution.py failed.
    exit /b 1
)

echo === Pipeline complete! Model saved to: %DIFF_OUTPUT% ===
