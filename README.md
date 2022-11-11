# paper_cleaner

## Download
- Clone this repo:
```bash
git clone https://github.com/min020/paper_cleaner
cd paper_cleaner
```
- model download:

  https://drive.google.com/file/d/1RG4by55nr12cUrr_RL_xCLtpbRI-Gg7c/view?usp=sharing

## Requirements
- install the requirements.txt

## Using paper_cleaner
- To cleaning an image use the followng command: 
```bash
python enhance.py ./model_path ./mess_image_path ./directory_to_cleaned_image
```
image:

<img src="https://github.com/min020/paper_cleaner/blob/9bbfb3d7e97b61ea31501b11d904726ef2e856e7/sample/mess_testpaper.png" width="246px" height="290px" alt="testpaper"></img>
<img src="https://github.com/min020/paper_cleaner/blob/9bbfb3d7e97b61ea31501b11d904726ef2e856e7/sample/mess_document.png" width="290px" height="290px" alt="document"></img><br/>

cleaned_image

<img src="https://github.com/min020/paper_cleaner/blob/9bbfb3d7e97b61ea31501b11d904726ef2e856e7/sample/clean_testpaper.png" width="246px" height="290px" alt="testpaper"></img>
<img src="https://github.com/min020/paper_cleaner/blob/9bbfb3d7e97b61ea31501b11d904726ef2e856e7/sample/clean_document.png" width="290px" height="290px" alt="document"></img>

## Training with your own data
- To train with your own data, place your mess images in the folder "data/A/" and corresponding ground-truth in the folder "data/B/". It is necessary that each mess image and its corresponding gt are having the same name (could have different extentions), also, the number images  should be the same in both folders.
- Command to train:
```bash
python train.py 
```
- Specifying the batch size and the number of epochs could be done inside the code.
