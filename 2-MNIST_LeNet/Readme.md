# Use LeNet to train a model on MNIST-Dataset

## Installation
1. Install necessary lib:
   - `pip install -r requirements.txt`

## Usage Guide

#### Examples

```bash
# This command trains a model with 10 epoch.
python train.py --epoch 10 

# This command trains a model with 10 epoch using a pretrained model.
python train.py --resume --checkpoint 1 --load_run Oct12_14-21-27_

# Evaluating the Trained Model
# This command loads the selected model for test dataset. 
python test.py --load_run Oct12_14-26-44_ --checkpoint 2

# Evaluating the Trained Model
# This command loads the selected model for local image. 
python eval.py --load_run Oct12_14-26-44_ --checkpoint 2

```
