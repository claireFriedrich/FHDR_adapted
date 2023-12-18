# A Feedback Network to reconstruct HDR images from LDR inputs


This repository is adapted from [the code](https://github.com/mukulkhanna/fhdr) linked to the [FHDR: HDR Image Reconstruction from a Single LDR Image using Feedback Network](https://arxiv.org/abs/1912.11463v1) authored by  Z. Khan, M. Khanna and S. Raman. <br>

The authors of the current repository are:

- Claire Alexandra Friedrich
- Céline Kalbermatten
- Adam Zinebi


The repository was created within the scope of a Machine Learning project during the course [CSS-433 Machine Learning](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) at [EPFL](https://www.epfl.ch/en/).

    
## Table of contents:

- [Abstract](#abstract)
- [Setup](#setup)
- [Files](#files)
- [Dataset](#dataset)
- [Training](#training)
- [Pretrained models](#pretrained-models)
- [Evaluation](#evaluation)
- [Acknowledgement](#acknowledgement)

## Abstract
> High dynamic range (HDR) image generation from a single exposure low dynamic range (LDR) image has been made possible due to the recent advances in Deep Learning. Various feed-forward Convolutional Neural Networks (CNNs) have been proposed for learning LDR to HDR representations. <br><br>
To better utilize the power of CNNs, the authors exploit the idea of feedback, where the initial low level features are guided by the high level features using a hidden state of a Recurrent Neural Network. Unlike a single forward pass in a conventional feed-forward network, the reconstruction from LDR to HDR in a feedback network is learned over multiple iterations. This enables to create a coarse-to-fine representation, leading to an improved reconstruction at every iteration. Various advantages over standard feed-forward networks include early reconstruction ability and better reconstruction quality with fewer network parameters. We design a dense feedback block and propose an end-to-end feedback network- FHDR for HDR image generation from a single exposure LDR image.

## Setup

### Pre-requisites

- Python3
- [PyTorch](https://pytorch.org/)
- GPU, CUDA, cuDNN
- [OpenCV](https://opencv.org)
- [PIL](https://pypi.org/project/Pillow/)
- [Numpy](https://numpy.org/)
- [argparse](https://pypi.org/project/argparse/)
- [scikit-image](https://scikit-image.org/)
- [tqdm](https://pypi.org/project/tqdm/)

**`requirements.txt`** is provided to install the necessary Python dependencies

```sh
pip install -r requirements.txt
```
**Note:** It is essential to install these exact dependencies to ensure that the model works.


## Files

- `split_data.py`
- `get_clear_sky.py`
- `dataloader.py`
- `model.py`
- `train.py`
- `test.py`
- `options.py`
- `vgg.py`
- `util.py`

### Description

The whole implementation of the project has been done in Python.

The file `split_data.py` creates a dataset in the structure needed for the training and testing of the model. More information can be found in the part about the [dataset](#dataset).

The file `get_clear_sky.py` creates a dataset only based on the clear sky images. More information can be found in the part about the [pretrained models](#training).

The file `dataloader.py` defines a custom HDR class that loads LDR and HDR images. It provides methods to transform the images into tensors and organize the into a dictionary.

The file `model.py` defines an Fast High Dynamic Range (FHDR) model consisting of initial feature extraction layers, a feedback block for iterative processing, layers for high-resolution reconstruction, and a final output transformation. The feedback block maintains the state across iterations using dilated residual dense blocks that preserve and update hidden states during each pass.

The file `train.py` is designed to train a model for HDR image reconstruction. It initializes the model, optimizes it using defined loss functions, does training and validation loops, saves intermediate results, and ultimately saves the trained model. Additionally, it plots the losses throughout the process.

The file `test.py`evaluates the trained HDR image model. It loads test data, applies the model to generate HDR images, saves the results, and computes evaluation metrics like PSNR and SSIM for the generated images compared to ground truth. The final results are printed.

The file `options.py` contains a class Options that defines and handles various settings and configurations used for training, debugging, and evaluation of the FHDR model. It uses the argparse module to define command-line arguments for different options like batch size, learning rate, number of epochs, GPU IDs, debugging flags, and testing options such as checkpoint paths and logging scores during evaluation. The parse() method parses these options and returns the parsed arguments.

The file `vgg.py` implements a VGG19 network for perceptual loss computation during training of HDR image generation models, using pre-trained layers to extract features and compute the loss.

The file `util.py` contains several utility functions including methods for checkpoint loading and saving, HDR image tonemapping, saving HDR and LDR images, updating learning rates and plotting losses. 


## Dataset

The dataset is expected to contain LDR (input) and HDR (ground truth) image pairs. The network is trained to learn the mapping from LDR images to their corresponding HDR ground truth counterparts.

The dataset should have the following folder structure - 

```
> dataset
    > train
        > HDR
            > hdr_image_1.hdr/exr
            > hdr_image_2.hdr/exr
            .
            .
        > LDR

            > ldr_image_1.jpg/png
            > ldr_image_2.jpg/png
            .
            .
    > test
        > HDR

            > hdr_image_1.hdr/exr
            > hdr_image_2.hdr/exr
            .
            .
        > LDR

            > ldr_image_1.jpg/png
            > ldr_image_2.jpg/png
            .
            .
```

- The full train and test datasets can be downloaded [here](https://drive.google.com/drive/folders/1KyE1_YEZJeJ_O8cztDCHOH0f19J_vnnb?usp=sharing)
- The clear-sky only dataset can be found [here](https://drive.google.com/drive/folders/12tjwbB6tuNMj8ZzcUc0grK9ltyrCgQ3v?usp=sharing)
- The pretrained model has been trained on 1700 256x256 LDR/HDR pairs generated by tiling 34 2560x1280 test LDR/HDR frames into 50 smaller frames (check in the report for details). 



### Create your own dataset

If you want to generate a dataset from your own images, order your LDR and HDR images according to the following folder structure:

```
> NAME_OF_THE_FOLDER_WITH_THE_DATA (put as data path)

    > raw_ref_video_hdr
        - contains the test HDR and LDR images in .hdr format

    > raw_video_png
        - contains the test LDR images in .png format

    > tile_ref_video_hdr
        - contains the training HDR and LDR images in .hdr format

    > tile_ref_video_png
        - contains the training HDR and LDR images in .png format
```

When your data is structured in the above ways, split your data by using the provided script: 
```sh
python3 split_data.py data_path
```

**Note:** `data_path` is the path (str) to the dataset on your local computer


## Training


After the dataset has been prepared, the model can be trained using:

```sh
python3 train.py
```
- Training results (LDR input, HDR prediction and HDR ground truth) are stored in the **`train_results`** directory.

The corresponding parameters/options for training have been specified in the **`options.py`** file and can be easily altered. They can be logged using -

```sh
python3 train.py --help
```
- **`--iter`** param is used to specify the number of feedback iterations for global and local feedback mechanisms (refer to paper/architecture diagram)
- Checkpoints of the model are saved in the **`checkpoints`** directory. (Saved after every 2 epochs by default)
- GPU is used for training. Specify GPU IDs using **`--gpu_ids`** param.
- The model takes around 80s per epoch so 5 hours to train on a dataset of 1700 images on a Tesla V100-PCIE-32GB GPU.

### Pretrained models


Three pre-trained models can be downloaded from the below-mentioned links. 

These models have been trained with the default options, on 256x256 size images for 200 epochs.

- [2-Iterations model from paper](https://drive.google.com/open?id=13vTGH-GVIWVL79X8NJra0yiguoO1Ox4V)
- [FHDR model trained on 1700 256x256 images with 200 epochs, only VGG loss](https://drive.google.com/file/d/1_Bp6kR56uttLXwW9IWdaiGZwmIoDIqlG/view?usp=drive_link)
- [FHDR model trained on 1700 256x256 images with 200 epochs, L1 + VGG loss](https://drive.google.com/file/d/1A80kL5PoNk37o_oCuKzAxVr-xu6g5yJ-/view?usp=sharing)
- [FHDR model trained on 500 clear sky 256x256 images with 200 epoch](https://drive.google.com/file/d/1E9aEWcUcdOWQuQqhrZ7dIzh0Xr-c01BT/view?usp=sharing)

<img src="https://github.com/claireFriedrich/FHDR_adapted/blob/main/img/mixed_loss_500_epochs.png" width="380" height="280" />

## Evaluation of the model

The performance of the network can be evaluated using: 

```sh
python3 test.py --ckpt_path /path/to/checkpoint
```

- Test results (LDR input, HDR prediction and HDR ground truth) are stored in the **`test_results`** directory.
- HDR images can be viewed using [OpenHDRViewer](https://viewer.openhdr.org). Or by installing [HDR + WCG Image Viewer](https://apps.microsoft.com/detail/9PGN3NWPBWL9?rtc=1&hl=fr-ch&gl=CH) on windows
- If checkpoint path is not specified, it defaults to `checkpoints/latest.ckpt` for evaluating the model.

**Note:** Inference can be done on CPU (45 min)

### Our testing results 
The 34 generated test HDR images for each of the above models can be found at the links below: 
- [Results of 2-Iterations model from paper](https://drive.google.com/drive/folders/1qmffl_CTiMT6DsWB6FUP7blZrgOJ1EJf?usp=sharing)
- [Results of FHDR model trained on 1700 256x256 images with 200 epochs, only VGG loss](https://drive.google.com/drive/folders/1TD6_lcl6PIMF_oM5_q7oQBqK91w_dvW_?usp=sharing)
- [Results of FHDR model trained on 1700 256x256 images with 200 epochs, VGG + L1 loss](https://drive.google.com/drive/folders/1itfrmDBN_RgWfKRNqkECfvQIcMqO1FE9?usp=sharing)
- [Results of FHDR model trained on 500 256x256 clear-sky images with 200 epoch](https://drive.google.com/drive/folders/1BxrDXyPI6w4A1OhEBBPtvc8oRD7xj8xj?usp=sharing)


### Acknowledgement

This project on HDR reconstruction was provided by the [Laboratory of Integrated Performance in Design (LIPID)](https://www.epfl.ch/labs/lipid/) at EPFL and supervised by Stephen Wasilewski and Cho Yunjoung.

We are grateful to Stephen Wasilewski, Cho Yunjoung and the entire team at LIPID for their support and guidance throughout this project. 

The code was adapted from the previously cited [repository](https://github.com/mukulkhanna/fhdr).

