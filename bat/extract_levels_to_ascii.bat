@echo off
REM Usage: extract_levels_to_ascii.bat <name> <count> <output_folder>
REM <name>          level name filter (passed to extract_mm2_bcd.py --name)
REM <count>         number of matching levels to extract (--name_count)
REM <output_folder> folder that will hold bcd/, json/, images/ and ascii/ subfolders
cd ..

set NAME=%~1
set COUNT=%~2
set OUTPUT=%~3

if "%NAME%"=="" (
    echo ERROR: Must provide a level name filter as the first argument.
    exit /b 1
)
if "%COUNT%"=="" (
    echo ERROR: Must provide the number of levels to extract as the second argument.
    exit /b 1
)
if "%OUTPUT%"=="" (
    echo ERROR: Must provide an output folder as the third argument.
    exit /b 1
)

for %%I in ("%OUTPUT%") do set "OUTPUT=%%~fI"

set BCD_DIR=%OUTPUT%\bcd
set JSON_DIR=%OUTPUT%\json
set IMAGES_DIR=%OUTPUT%\images
set ASCII_DIR=%OUTPUT%\ascii

echo === Step 1: Extracting %COUNT% level^(s^) matching "%NAME%" ===
python extract_mm2_bcd.py --name "%NAME%" --name_count %COUNT% --output_dir "%BCD_DIR%" --skip_3dworld --skip_items --skip_subworld_items
if %ERRORLEVEL% neq 0 ( echo ERROR: extract_mm2_bcd.py failed. & exit /b 1 )

echo === Step 2: Converting .bcd files to .json and images ===
pushd toost_stuff
python batch_convert.py "%BCD_DIR%" -o "%JSON_DIR%" --images-output "%IMAGES_DIR%"
if %ERRORLEVEL% neq 0 ( popd & echo ERROR: batch_convert.py failed. & exit /b 1 )
popd

echo === Step 3: Converting .json files to ASCII ===
python mm2_json_to_ascii.py "%JSON_DIR%" "%ASCII_DIR%"
if %ERRORLEVEL% neq 0 ( echo ERROR: mm2_json_to_ascii.py failed. & exit /b 1 )

echo === Done! ===
echo   BCDs:   %BCD_DIR%
echo   JSON:   %JSON_DIR%
echo   Images: %IMAGES_DIR%
echo   ASCII:  %ASCII_DIR%
