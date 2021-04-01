import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

# load dataset images
train_data = datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor())
train, test = random_split(train_data, [55000, 5000])
train_loader = DataLoader(train, batch_size=32)
test_loader = DataLoader(test, batch_size=32)
Loders = (train_loader, test_loader)

# hyper-parameters for this model
device = "cpu"  # decide between cpu or gpu
n_epochs = 10
n_classes = 26
lr = 0.001  # learning rate
train_samples = len(train_loader)
test_samples = len(test_loader)
input_layer_nurons = 784  # 28x28 pixels of image representation
hidden_layer = 100

checkdata = iter(train_loader)
img, lab = next(checkdata)
print(img.shape, lab.shape)
plt.imshow(img[1][0], cmap="gray")
plt.show()

# define the model - create layers.

model = nn.Sequential(
    nn.Linear(in_features=input_layer_nurons, out_features=hidden_layer),
    nn.ReLU(),
    nn.Linear(in_features=hidden_layer, out_features=hidden_layer),
    nn.ReLU(),
    nn.Linear(in_features=hidden_layer, out_features=n_classes)
)
model = model.to(device)
# optimiser
params = model.parameters()  # return dictinary
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # learning rate

# Define my loss
loss = nn.CrossEntropyLoss()

# my Training loop
n_epochs = 5
for epoch in range(n_epochs):
    for loader in Loders:
        losses = list()
        total_train = 0
        acc_train = 0
        for step, (images, labels) in enumerate(loader):

            # img: batch size * 1 * 28 * 28 - 1 for gray-scale image
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            # b= img.size(0)
            # img = img.view(b, -1) # matrix with b rows and 28 columns

            # step 1: Forward pass: Compute Loss
            outputs = model(images)

            # step2 - compute the objective function
            loss_func = loss(outputs, labels)
            # losses accumulate
            losses.append(loss_func.item())

            if loader is train_loader:
                # step 3 - cleaninig gradients
                optimizer.zero_grad()  # zeroes all gradians from previous steps
                # params,grad,zero_()

                # step 4  - compute the patial derivatives of J wrt params
                loss_func.backward()  # compute the new gradians
                # params,frad,add_(dJ/dparams)

                # step 5 - Backward pass: Compute dLoss / dWeights
                optimizer.step()
                # with torch.no_grad(): params = params - eta * params.grad

            # accuracy calculation:
            total_train += labels.size(0)
            # tensor array of the highest value neuron for each image
            # in the batch
            _, output_neurons = torch.max(outputs.data, 1)
            # comparison between the output_neurons and the
            # labels.data to accumulate success prediction over epoch
            acc_train += output_neurons.eq(labels.data).sum().item()

            # show progress:
            if loader is train_loader and step == train_samples - 1:
                # final calculation for loss average and accuracy percentage
                loss_avg = torch.tensor(losses).mean()
                accuracy = 100 * acc_train / total_train
                print("Train: Epoch {}/{}, Samples: {}, Loss: {:.4f}, Accuracy: {:.2f}%".format(epoch + 1, n_epochs,
                                                                                                train_samples, loss_avg,
                                                                                                accuracy))
            if loader is test_loader and step == test_samples - 1:
                # final calculation for loss average and accuracy percentage
                loss_avg = torch.tensor(losses).mean()
                accuracy = 100 * acc_train / total_train
                print("Test: Epoch {}/{}, Samples: {}, Loss: {:.4f}, Accuracy: {:.2f}%".format(epoch + 1, n_epochs,
                                                                                               test_samples, loss_avg,
                                                                                               accuracy))
