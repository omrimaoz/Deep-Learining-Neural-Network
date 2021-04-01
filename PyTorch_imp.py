import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

train_data = datasets.MNIST('data', train = True, download = False, transform=transforms.ToTensor())
train, val = random_split(train_data, [55000, 5000])
train_loader = DataLoader(train, batch_size = 32)
val_loader = DataLoader(val, batch_size=32)


for batch in train_loader:
    plt.grid(False)
    plt.imshow(batch[0], cmap=plt.cm.binary)  # plt.cm.binary for gray scale
    plt.show()
    break


model = nn.Sequential(
    nn.Linear(28 * 28, 64),
    nn.ReLU(),
    nn.Linear(64,64),
    nn.ReLU(),
    nn.Linear(64, 26)
)

#optimiser
params = model.parameters() # return dictinary
optimizer = torch.optim.SGD(model.parameters(),lr=100) #learning rate

#Define my loss
loss = nn.CrossEntropyLoss()

#my Training loop
n_epochs = 5
for epoch in range(n_epochs):
    for batch in train_loader:
        img, label = batch

        #img: batch size * 1 * 28 * 28 - 1 for gray-scale image
        b= img.size(0)
        img = img.view(b, -1) # matrix with b rows and 28 columns

        #step 1: Forward pass: Compute Loss
        logit = model(img)

        #step2 - compute the objective function
        J = loss(logit, label)

        #step 3 - cleaninig gradients
        model.zero_grad() # zeroes all gradians from previous steps
        #params,grad,zero_()

        #step 4  - compute the patial derivatives of J wrt params
        J.backward() # compute the new gradians
        #params,frad,add_(dJ/dparams)

        step 5 - Backward pass: Compute dLoss / dWeights
        optimizer.step()
        #with torch.no_grad(): params = params - eta * params.grad


