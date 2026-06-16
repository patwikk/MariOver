@echo off
REM Usage: train-mario-maker.bat [seed]
REM [seed] is optional, defaults to 0
REM Trains a text-conditional diffusion model on Mario Maker levels
REM using simple presence-based captions (MarioMaker_create_ascii_captions.py).
REM Run prepare-mario-maker.bat first to build and split the dataset.
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set GAME=MM
set TYPE=regular

set MLM_OUTPUT=%GAME%-MLM-%TYPE%%SEED%
set DIFF_OUTPUT=%GAME%-conditional-%TYPE%%SEED%

REM Used to auto-answer "y" to train_diffusion.py's resume-from-checkpoint prompt.
REM A redirected file (rather than a pipe) keeps %ERRORLEVEL% checks working afterward.
set YES_FILE=%TEMP%\mariover_yes.txt
echo y> "%YES_FILE%"

python train_mlm.py --epochs 300 --save_checkpoints --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-validate.json --test_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-test.json --pkl datasets\%GAME%_Tokenizer-%TYPE%.pkl --output_dir %MLM_OUTPUT% --seed %SEED%
python train_diffusion.py --game %GAME% --save_image_epochs 1000 --augment --text_conditional --output_dir "%DIFF_OUTPUT%" --num_epochs 500 --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --pkl datasets\%GAME%_Tokenizer-%TYPE%.pkl --mlm_model_dir %MLM_OUTPUT% --seed %SEED% < "%YES_FILE%"
call bat\run_diffusion_multi.bat %DIFF_OUTPUT% %TYPE% %GAME%
call bat\evaluate_caption_adherence_multi.bat %DIFF_OUTPUT% %TYPE% %GAME%
python evaluate_tile_distribution.py --model_path %DIFF_OUTPUT% --num_tiles 17 --tileset extended_tiles.json



