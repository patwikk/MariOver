@echo off
REM Usage: prepare-mario-maker.bat <input> [seed]
REM <input> is a path to a .txt ASCII level file or a folder of .txt files
REM [seed] is optional, defaults to 0
cd ..

set INPUT=%1
if "%INPUT%"=="" (
    echo ERROR: Must provide input path as first argument.
    exit /b 1
)

set SEED=%2
if "%SEED%"=="" set SEED=0

set GAME=MM
set TYPE=regular
set TILESET=extended_tiles.json

set RAW_OUTPUT=datasets\%GAME%_Levels-%TYPE%.json
set CAPTIONED_OUTPUT=datasets\%GAME%_LevelsAndCaptions-%TYPE%.json

python build_dataset_with_ascii.py --input_file %INPUT% --output %RAW_OUTPUT% --tileset %TILESET% --sliding_window --stride 20 --convert_to_extended
python MarioMaker_create_ascii_captions.py --dataset %RAW_OUTPUT% --tileset %TILESET% --output %CAPTIONED_OUTPUT%
python split_mario_maker_data.py --json %CAPTIONED_OUTPUT% --seed %SEED%
python tokenizer.py save --json_file datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --pkl_file datasets\%GAME%_Tokenizer-%TYPE%.pkl
python create_mario_maker_random_captions.py --json %CAPTIONED_OUTPUT% --output datasets\%GAME%_RandomTest-%TYPE%.json
