'''
Use GAN to train models on MNIST-Dataset
'''
import os
from datetime import datetime
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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

log_root = os.path.join(ROOT_DIR, '3-MNIST_GAN', 'log')

args = get_args()

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Download MNIST dataset
train_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST('./dataset', train=False, transform=trans, download=False)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

device = torch.device(args.device)

loss = nn.BCELoss()

# if we want to resume training
if args.resume:
    resume_path_G = get_load_path_G(log_root, load_run=args.load_run_G, checkpoint=args.checkpoint_G)
    resume_path_D = get_load_path_D(log_root, load_run=args.load_run_D, checkpoint=args.checkpoint_D)
    print(f"Loading model from G: {resume_path_G}")
    print(f"Loading model from D: {resume_path_D}")
    net_D = torch.load(resume_path_D)
    net_G = torch.load(resume_path_G)
    optimizer_D = optim.Adam(net_D.parameters(), lr=args.lr_D)
    optimizer_G = optim.Adam(net_G.parameters(), lr=args.lr_G)
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_')
    print("load model successfully")
else:
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_')
    net_D = D(input_dim=784).to(device)
    net_G = G(input_dim=100, output_dim=784).to(device)
    optimizer_D = optim.Adam(net_D.parameters(), lr=args.lr_D)
    optimizer_G = optim.Adam(net_G.parameters(), lr=args.lr_G)
    
# train
for epoch in range(args.epoch):
    for i, (img, _) in enumerate(train_loader):
        
        # get the img and fake_img
        img = img.view(-1, 784).to(device)# flatten the image to 784 dim
        noise = torch.randn(img.size(0), 100).to(device)
        fake_img = net_G(noise)
        
        # define the labels
        real_label = torch.ones(img.size(0), 1).to(device)
        fake_label = torch.zeros(img.size(0), 1).to(device)

        # train the discriminator
        optimizer_D.zero_grad()
        output = net_D(img)
        loss_real = loss(output, real_label)
        loss_real.backward()
        output = net_D(fake_img.detach())
        loss_fake = loss(output, fake_label)# hope the discriminator can distinguish the fake_img
        loss_fake.backward()
        optimizer_D.step()

        # train the generator
        optimizer_G.zero_grad()
        output = net_D(fake_img)
        loss_G = loss(output, real_label)# hope the generator can generate the img that the discriminator can't distinguish
        loss_G.backward()
        optimizer_G.step()

        if i % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}'.format(
                epoch+1, args.epoch, i, len(train_loader), loss_real+loss_fake, loss_G
            ))
    # every 10 epoch, save the model
    if (epoch+1) % 10 == 0 and epoch != 0:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        torch.save(net_G.state_dict(), os.path.join(log_dir, 'generator_model_{}.pth'.format(epoch+ 1 + args.checkpoint_G)))
        torch.save(net_D.state_dict(), os.path.join(log_dir, 'discriminator_model_{}.pth'.format(epoch+ 1+ args.checkpoint_D)))
        print("save model successfully")
if epoch < 10:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    torch.save(net_G.state_dict(), os.path.join(log_dir, 'generator_model_{}.pth'.format(epoch+ 1+ args.checkpoint_G)))
    torch.save(net_D.state_dict(), os.path.join(log_dir, 'discriminator_model_{}.pth'.format(epoch+ 1+ args.checkpoint_D)))
    print("save model successfully")





