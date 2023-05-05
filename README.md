1. Create a new conda environment and install the dino-requirements.txt to it.
2. Create a new folder called weights and download the weights to that folder by using this command: wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
3. Use this command to run the file:
python3 annotator.py --config_file groundingdino/config/GroundingDINO_SwinT_OGC.py --checkpoint_path weights/groundingdino_swint_ogc.pth --image_dir /path/to/img/dir --text_prompt "classes(separated by .)"  --save_inf True(if you want to save the inferred images) --output /path/to/save/inffered/images
