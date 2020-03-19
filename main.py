import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor()]
)

BATCH_SIZE = 32

train_set = dset.FashionMNIST(root='./data', train=True, 
                                download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, 
                                            shuffle=True, num_workers=2)

test_set = dset.FashionMNIST(root='./data', train=False, 
                                download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, 
                                            shuffle=False, num_workers=2)

