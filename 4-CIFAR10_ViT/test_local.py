'''
test the model on LOCAL CIFAR-10 dataset (balanced dataset)
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

test_dataset = CustomDataset(root_dir='../CIFAR10_balanced/', transform=transform)

device = torch.device(args.device)

test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)


vit = ViT(image_size=32, patch_size=4, num_classes=10, dim=128, depth=6, heads=8, mlp_dim=256, pool='cls', channels=3).to(device)
resume_path = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
print(f"Loading model from: {resume_path}")
vit.load_state_dict(torch.load(resume_path))
vit.eval()

loss_fn = nn.CrossEntropyLoss()

size = len(test_loader.dataset)
num_batches = len(test_loader)
vit.eval()
test_loss, correct = 0, 0
with torch.no_grad():
    for X, y in tqdm(test_loader):
        X, y = X.to(device), y.to(device)
        pred = vit(X)
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
test_loss /= num_batches
correct /= size
print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")




