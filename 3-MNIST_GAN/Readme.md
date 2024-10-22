# Use GAN to train models on MNIST-Dataset

## Installation
1. Install necessary lib:
   - `pip install -r requirements.txt`

## Usage Guide

#### Examples

```bash
# This command trains a model with 10 epoch.
python train.py --epoch 10 

# This command trains a model with 10 epoch using a pretrained model.
python train.py --epoch 10 --resume --checkpoint_G 19 --load_run_G Oct21_20-02-30_ --checkpoint_D 19 --load_run_D Oct21_20-02-30_

# Generating the image using the Trained Model
# This command loads the selected model for local image. 
python generate.py --load_run_G Oct21_19-52-50_ --checkpoint_G 19

```
