'''
Use LeNet to train a model on MNIST-Dataset
'''
import os
from datetime import datetime
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from network import LeNet
from torch import nn
from utils import *

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

log_root = os.path.join(ROOT_DIR, '2-MNIST_LeNet', 'log')

args = get_args()

trans = transforms.ToTensor()

# MNIST数据集加载
train_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST('./dataset', train=False, transform=trans, download=False)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

device = torch.device(args.device)

net = LeNet().to(device)

loss_fn = nn.CrossEntropyLoss()

epoch = args.epoch
total_train_step = 0

# if resume, load the model
if args.resume:
    resume_path = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
    print(f"Loading model from: {resume_path}")
    net = torch.load(resume_path)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_')
    print("load model successfully")
else:
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_')
    net = LeNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
for i in range(epoch):
    print("================Traning Epoches:{}=================".format(i+1+args.checkpoint))
    for data in train_loader:
        image , targets = data
        image = image.to(device)
        targets = targets.to(device)

        output = net(image)
        loss = loss_fn(output,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 1000 == 0:
            print("Step:{},Loss:{}".format(total_train_step,loss))

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
torch.save(net, os.path.join(log_dir, 'model_{}.pt'.format(epoch+args.checkpoint)))




