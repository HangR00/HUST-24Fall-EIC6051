'''
test the model on CIFAR-10 dataset
'''
import os
from datetime import datetime
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from network import ViT
from tqdm import tqdm
from utils import *

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

log_root = os.path.join(ROOT_DIR, '4-CIFAR10_ViT', 'log')

args = get_args()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = torch.device(args.device)

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