# MariOver
The backend of [The Mario Maker 2 API](https://tgrcode.com/mm2/docs/). Hey Nintendo, it's MariOver. (Thanks TGR!)

# Setting up
0. Run `pip install -r requirements.txt`
(Not much for now....)


# Useful Commands

**Run the HuggingFace visualizer**
python mm2_viewer.py

**Run the dataset visualizer**
python ascii_browser.py
(For load dataset you need to load your selected dataset and the smb.json outside the folder)

**Creating a dataset:**
python build_dataset.py --keyword (keyword) --max_levels (amount of levels to look at) --output (dataset_name).json

**Creating a dataset with captions:**
python build_dataset.py --keyword (keyword) --max_levels (amount of levels to look at) --output (dataset_name).json --caption --exclude_upside_down_pipes --seed 0

**Train unconditional diffusion model:**
python train_diffusion.py --json datasets/(dataset file) --game Mario --output_dir (output_folder)

**Train captioning model:** 
python train_mlm.py --epochs 300 --save_checkpoints --json datasets\(dataset_name)_captioned.json --val_json datasets\(dataset_name)_captioned-validate.json --test_json datasets\d(dataset_name)_captioned-test.json --pkl datasets\(dataset_name)_Tokenizer-regular.pkl --output_dir dataset-MLM

**Training the (captioned) diffusion model:**
python train_diffusion.py --augment --text_conditional --output_dir (dataset_name)-conditional --num_epochs 500 --json datasets\(dataset_name)_captioned-train.json --val_json datasets\(dataset_name)_captioned-validate.json --pkl datasets\(dataset_name)_Tokenizer-regular.pkl --mlm_model_dir dataset-MLM --plot_validation_caption_score --seed 0


**Running the unconditional diffusion model:**
python run_diffusion.py --model_path (training_folder) --num_samples (number of samples) --output_dir (training_folder)_SAMPLES --save_as_json

**Running captioned diffusion with GUI:**
python interactive_tile_level_generator.py --model_path (generated_dataset) --load_data datasets/Mar1and2_LevelsAndCaptions-regular.json

**Running MarioMaker_create_ascii_captions.py:**
python MarioMaker_create_ascii_captions.py --dataset (input.json) --tileset (tileset.json) --output (output.json)