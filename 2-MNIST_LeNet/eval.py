'''
Load the trained model and evaluate the model on the local handwritten image
'''
import os
import datetime
import numpy as np
import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from network import LeNet
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
from utils import *

args = get_args()

transform = transforms.Compose([
    transforms.Resize((28,28)),   
    transforms.ToTensor(),  
])
device = torch.device(args.device)

# load the image
img = Image.open(('./test_dataset/handwriting_2.jpg')).convert('L')
img = img.point(lambda p: 0 if p > 128 else 255)

# angle
angle = 45
# scale
width_scale = 0.9
height_scale = 0.3
# translate
translate = 10

img_scaled = scale_image(img, width_scale, height_scale)
img_rotated = rotate_image(img, angle)  
img_translated = translate_image(img, translate, 'left')

input_image = transform(img)
input_image = input_image.unsqueeze(0) 
input_image = input_image.to(device)

log_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '2-MNIST_LeNet', 'log')

resume_path = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)

print(f"Loading model from: {resume_path}")

model = torch.load(resume_path)
model.eval()
with torch.no_grad():
    output = model(input_image)
    # print(output)
    pred = output.data.max(1, keepdim=True)[1]
    print('pred:', output.data)
    predicted_label = pred.item()  
    print("Predicted label:", predicted_label)
    