import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from torch.autograd import Variable
from PIL import Image
from collections import OrderedDict
from collections import namedtuple
from itertools import product, chain
from model import CNN

transform = transforms.Compose([transforms.ToTensor()])

# Inputs
BATCH_SIZE = 100
learning_rate = .001
epochs = 5

## Training and test sets

train_set = dset.FashionMNIST(
    root = './data', 
    train = True, 
    download = True, 
    transform = transform
)

train_loader = torch.utils.data.DataLoader(
    train_set, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)

test_set = dset.FashionMNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

test_loader = torch.utils.data.DataLoader(
    test_set, 
    batch_size=BATCH_SIZE, 
    shuffle=False
)

def output_label(label):
    '''
    A Method that return the name of a class 
    for the label number '''

    output_mapping = {
        0 : 'T-Shirt/Top',
        1 : 'Trouser',
        2 : 'Pullover',
        3 : 'Dress',
        4 : 'Coat',
        5 : 'Sandal',
        6 : 'Shirt',
        7 : 'Sneaker',
        8 : 'Bag',
        9 : 'Ankle Boot'
    }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]

# # Plots some images and labels
# fig = plt.figure(figsize=(8,8))
# columns = 4
# rows = 5
# for i in range(1, columns*rows+1):
#     img_xy = np.random.randint(len(train_set))
#     img = train_set[img_xy][0][0,:,:]
#     fig.add_subplot(rows, columns, i)
#     plt.title(labels[train_set[img_xy][1]])
#     plt.axis('off')
#     plt.imshow(img, cmap='gray')
# plt.show()

## Using the GPU if available. Else use the CPU
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

## Using the model
model = CNN()
model.to(device)

## Criterion
error = nn.CrossEntropyLoss()

## Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

count = 0
loss_list = []
iteration_list = []
accuracy_list = []

predictions_list = []
labels_list = []

for epoch in range(epochs):
    for images, labels in train_loader:
        # Transferring images and labels to GPU (if possible)
        images, labels = images.to(device), labels.to(device)

        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)

        # Forward pass
        outputs = model(train)
        loss = error(outputs, labels)

        # Init a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()

        # Propaganating the error backward
        loss.backward()

        # Optimizing the parameters
        optimizer.step()

        count += 1

        # Testing the model 
        if count % 50 == 0:
            total = 0
            correct = 0

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)

                test = Variable(images.view(100, 1, 28, 28))

                outputs = model(test)

                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()

                total += len(labels)

            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

        if not (count % 500):
            print('Iteration: {}, Loss: {}, Accuracy: {}%'
                .format(count, loss.data, accuracy))

## Prepare labels and predictions for the confusion matrix
predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]
labels_l = [labels_list[i].tolist() for i in range(len(labels_list))]
predictions_l = list(chain.from_iterable(predictions_l))
labels_l = list(chain.from_iterable(labels_l))

## Creating and printing the confusion matrix
metrics.confusion_matrix(labels_l, predictions_l)
print("Classification report for CNN :\n%s\n"
      % (metrics.classification_report(labels_l, predictions_l)))

## Save the model
torch.save(model.state_dict(), 'model_final.pth')