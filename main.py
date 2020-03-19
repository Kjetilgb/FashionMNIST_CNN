import cv2
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable
from PIL import Image

transform = transforms.Compose(
    [transforms.RandomResizedCrop(64),
    transforms.ToTensor()]
)

BATCH_SIZE = 100

train_set = dset.FashionMNIST(root='./data', train=True, 
                                download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, 
                                            shuffle=True)

test_set = dset.FashionMNIST(root='./data', train=False, 
                                download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, 
                                            shuffle=False)

def img_show(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(20,20))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(train_loader)
images, labels = dataiter.next()

img_show(torchvision.utils.make_grid(images))
# classes = trainset.classes
## input shape (width=28, height=28, depth=1)