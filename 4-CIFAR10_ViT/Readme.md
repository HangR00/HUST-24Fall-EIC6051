# Use ViT to train a model on CIFAR-10 Dataset

## Installation
1. Install necessary lib:
   - `pip install -r requirements.txt`

## Usage Guide

#### Attention
Two local datasets are provided on the last folder, which represent the imbalanced dataset and the balanced dataset respectively.

Two files should be unzipped to local folder before running the code.

#### Examples

```bash
# This command trains a model with 10 epoch && 128 batch_size on CIFAR-10 Dataset.
python train.py --epoch 10 --batch_size 128

# This command trains a model with 10 epoch && 128 batch_size on LOCAL Imbalanced CIFAR-10 Dataset.
python train_local.py --epoch 10 --batch_size 128

# This command trains a model with 10 epoch using a pretrained model.
python train.py --resume --checkpoint 50 --load_run Oct22_15-04-32_ 

# Evaluating the Trained Model
# This command loads the selected model for testing CIFAR-10 Dataset. 
python test.py --load_run Oct22_15-04-32_  --checkpoint 50

# Evaluating the Trained Model
# This command loads the selected model for testing LOCAL Imbalanced CIFAR-10 Dataset. 
python test_local.py --load_run Oct22_15-04-32_ --checkpoint 50

```
