# ROAM
A weakly supervised computational pathology method for clinical-grade auxiliary diagnosis and molecular-morphological biomarker discovery of gliomas

<img src="./docs/ROAM.png" width="800px" aligh="center">

# Pre-requisites:
Python (3.8.13), h5py (3.6.0), openslide (version 3.4.1), opencv (version 4.5.5), pillow (version 6.2.1), Pytorch (version 1.12.1), scikit-learn (version 1.0.2), matplotlib (version 3.5.2) and seaborn (version 0.11.2).

# Instructions
We provide the complete process of using ROAM. You can modify related config files and parameters to train ROAM with your own dataset.
## Data prepare 
The first step is to prepare training dataset. The WSI data should be first segmented to several patches (ROI in ROAM, size is 2048*2048). Patches are then cropped from each ROI and put into pre-trained model to extract features. All the features of patches within a WSI form a bag for training.

WSI data and corresponding detailed information (.csv file) shoule be ready.
The format of digitized whole slide image data should be standard formats (.svs,.tiff etc.) that can be read with openslide (version 3.4.1).

Here is an example of directory for WSI data.
```bash
DATA_DIRECTORY
      |____ slide1.tiff
      |____ slide2.tiff
      |____ ...
```

The corresponding information data of WSI (.csv) should include key information such as `slide_id`,`path`,`label`. Here is an example:

|slide_id|path|label|level|
|-|-|-|-|
|965c821d05bbec8|/images/202007/965c821d05bbec8.tif|2|x20|

`level` is the magnification level of the slide. More information can be added to facilitate the process of the data. We provide an example csv file in `./data_prepare/data_csv/`.


* WSI segmentation and patching

The first step is to segment the tissue and crop patches from the tissue region. We referenced CLAM's WSI processing method. CLAM provide a robust WSI segmentation and patching implementation. You can refer to [CLAM](https://github.com/mahmoodlab/CLAM) for more detailed information.

You can segment the tissue region and crop patches by running the following command in `./data_prepare/` directory:
```shell
python create_patches_fp.py --source DATA_DIRECTORY --datainfo data_csv/data_info.csv --patch_size 2048 --step_size 2048 --save_dir PATCH_DIRECTORY --patch_level 0
```
The description of parameters is listed as follow:

`--source`

The above commond will segment all WSIs in `data_info.csv`, crop patches within the tissue regions with the size of 2048 and generate the following folder structure:
```bash
PATCH_DIRECTORY
      |____ masks
            |__ slide1.jpg
            |__ slide2.jpg
            |__ ...
      |____ patches
            |__ slide1.h5
            |__ slide2.h5
            |__ ...
      |____ stitches
            |__ slide1.jpg
            |__ slide2.jpg
            |__ ...
      |____ process_list_autogen.csv
```
The `masks` folder contains the segmentation results (one image per slide).
The `patches` folder contains arrays of extracted tissue patches from each slide (one .h5 file per slide, where each entry corresponds to the coordinates of the top-left corner of a patch)
The `stitches` folder contains downsampled visualizations of stitched tissue patches (one image per slide) 





## Training with ROAM


