'''
load the generator model and generate the image
'''
import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from network import D, G
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils import *

def save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize, range=range, scale_each=scale_each, pad_value=pad_value)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)
    
    
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
log_root = os.path.join(ROOT_DIR, '3-MNIST_GAN', 'log')

args = get_args()
device = torch.device(args.device)

net_G = G(input_dim=100, output_dim=784).to(device)
resume_path_G = get_load_path_G(log_root, load_run=args.load_run_G, checkpoint=args.checkpoint_G)
print(f"Loading model from: {resume_path_G}")
net_G.load_state_dict(torch.load(resume_path_G))
net_G.eval()

num = 64
z = torch.randn(num, 100).to(device)
fake_img = net_G(z)
fake_img = fake_img.view(-1, 1, 28, 28)
save_image(fake_img, os.path.join(log_root, 'fake_img.png'), nrow=8, normalize=True)
print("generate the image successfully")




