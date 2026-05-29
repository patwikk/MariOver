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

**Running the diffusion model:**
python run_diffusion.py --model_path (training_folder) --num_samples (number of samples) --output_dir (training_folder)_SAMPLES --save_as_json
