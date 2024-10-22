'''
train the model on LOCAL CIFAR-10 dataset (imbalanced dataset)
'''
import os
from datetime import datetime
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from network import ViT
from tqdm import tqdm
from utils import *

class CustomDataset(Dataset):
     def __init__(self, root_dir, transform=None):
         self.root_dir = root_dir
         self.transform = transform
         self.images = []
         self.labels = []

         for label in range(10):
             folder_path = os.path.join(root_dir, str(label))
             for file_name in os.listdir(folder_path):
                 file_path = os.path.join(folder_path, file_name)
                 self.images.append(file_path)
                 self.labels.append(label)

     def __len__(self):
         return len(self.images)

     def __getitem__(self, idx):
         image_path = self.images[idx]
         image = Image.open(image_path).convert('RGB')
         label = self.labels[idx]

         if self.transform:
             image = self.transform(image)

         return image, label

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

log_root = os.path.join(ROOT_DIR, '4-CIFAR10_ViT', 'log')

args = get_args()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = CustomDataset(root_dir='../CIFAR10_imbalanced/', transform=transform)

device = torch.device(args.device)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

vit = ViT(image_size=32, patch_size=4, num_classes=10, dim=128, depth=6, heads=8, \
            mlp_dim=256, pool='cls', channels=3).to(device)

loss_fn = nn.CrossEntropyLoss()

# if resume, load the model
if args.resume:
    resume_path = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
    print(f"Loading model from: {resume_path}")
    vit = torch.load(resume_path)
    optimizer = torch.optim.Adam(vit.parameters(), lr=args.lr)
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_')
    print("load model successfully")
else:
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_')
    vit = ViT(image_size=32, patch_size=4, num_classes=10, dim=128, depth=6, heads=8, \
            mlp_dim=256, pool='cls', channels=3).to(device)
    optimizer = torch.optim.Adam(vit.parameters(), lr=args.lr)

for i in range(args.epoch):
    print("================Traning Epoches:{}=================".format(i+1+args.checkpoint))
    for images, labels in tqdm(train_loader):
        
        images = images.to(device)
        labels = labels.to(device)
        pred = vit(images)
        loss = nn.CrossEntropyLoss()(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print('Loss:', loss.item())
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    if (i+1) % 10 == 0 and i != 0:
        torch.save(vit, os.path.join(log_dir, 'model_{}.pt'.format(i+ 1+ args.checkpoint)))

        
        
        