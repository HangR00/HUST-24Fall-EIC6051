'''
Test the model on the test dataset
'''
import os
import datetime
import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from network import LeNet
from PIL import Image
import numpy
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from utils import *

args = get_args()

trans = transforms.ToTensor()

test_dataset = torchvision.datasets.MNIST('./dataset', train=False, transform=trans, download=False)
test_loader = DataLoader(test_dataset, batch_size= args.batch_size)

device = torch.device(args.device)

prediction = []
labels = []
test_losses = []
correct = 0
test_loss = 0

log_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '2-MNIST_LeNet', 'log')

resume_path = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
print(f"Loading model from: {resume_path}")

model = torch.load(resume_path)
model.eval()

with torch.no_grad():
    for image, target in test_loader:
        image = image.to(device)
        target = target.to(device)
        output = model(image)
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        prediction.extend(pred.cpu().numpy())
        labels.extend(target.cpu().numpy())

test_loss /= len(test_loader.dataset)
test_losses.append(test_loss)
print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{}({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. *correct / len(test_loader.dataset)
    ))

accuracy = accuracy_score(labels, prediction)
print('Accuracy:', accuracy)
