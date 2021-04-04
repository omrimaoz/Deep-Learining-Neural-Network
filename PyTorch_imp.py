# imports:
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from Decompress_Dataset import decompress_zlibFile


def create_train_test_loaders(data):
    # define numpy arrays
    labels = np.empty(104000, dtype=np.int64)
    for i in range(26):
        labels[i * 4000:(i + 1) * 4000] = np.full(4000,
                                                  i)  # fill array with labels - 10400 images divided to 26 letters

    # convert to torch.Tensor arrays while preserve Numpy dtype
    tensor_images = torch.Tensor(data.copy())
    tensor_labels = torch.from_numpy(labels)  # Tensor(labels,dtype='int64')

    # create Tensor Dataset - bind image to it's label
    dataset = TensorDataset(tensor_images, tensor_labels)

    # random split the dataset with constant - 104000 * 0.875 = 91000
    train, test = random_split(dataset, [int(data.shape[0] * 0.875), int(data.shape[0] * 0.125)])

    # create loaders with fixed batch size
    train_loader = DataLoader(train, batch_size=32)
    test_loader = DataLoader(test, batch_size=32)

    # stage screen massage:
    print("Successfully prepared data\n")

    return train_loader, test_loader


def main():
    # # load dataset images
    dataset = decompress_zlibFile("compressed_Dataset")  # dataset is 3d numpy array flatten into 2d array.
    Loaders = create_train_test_loaders(dataset)

    # model hyper-parameters
    device = "cpu"  # decide between cpu or gpu
    n_epochs = 4  # define how many time the model see the train images from the data
    n_classes = 26  # number of letters - will be the the number of output neurons
    lr = 0.001  # learning rate
    train_samples = len(Loaders[0])
    test_samples = len(Loaders[1])
    input_layer_neurons = 784  # 28x28 pixels of image representation
    hidden_layer = 128  # number of neurons in the hidden layer

    # label names
    letter_digit = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
                    ]

    # define the model - create layers.
    # layer 1 - input layer, layer inputs are the 28x28 pixels of the image flatten
    # into a 1 dimension numpy array. Applies a linear transformation to the incoming data.
    # layer 2,4 - a fully connected layer with 128 neurons and an activation function
    # - 'ReLU' - rectified linear unit.
    # layer 3,5 - a Linear layer, that means a fully connected layer with 128 neurons
    # that applies a linear transformation.

    model = nn.Sequential(
        nn.Linear(in_features=input_layer_neurons, out_features=hidden_layer),
        nn.ReLU(),
        nn.Linear(in_features=hidden_layer, out_features=hidden_layer),
        nn.ReLU(),
        nn.Linear(in_features=hidden_layer, out_features=n_classes)
    )
    model = model.to(device)  # make sure model is compatible for the cpu

    # optimiser - define optimizer object that will hold the current state and will update the parameters based on
    # the computed gradients.
    # optim.Adam is an optimization algorithms that works very well with this sort of classification
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # learning rate

    # Define model loss
    loss = nn.CrossEntropyLoss()

    # stage screen massage:
    print("Successfully defines model\n")
    print("Start training and testing the model")

    # Training loop - model training and testing with the data
    for epoch in range(n_epochs):  # loop over epochs
        for loader in Loaders:  # train and then test
            losses = list()  # empty list of losses to get losses average
            total_train = 0  # variables to calculate accuracy percentage
            acc_train = 0  # variables to calculate accuracy percentage
            for step, (images, labels) in enumerate(loader):  # loop over loader batch

                # img: batch size * 1 * 28 * 28 - 1 for gray-scale image
                images = images.reshape(-1, 28 * 28).to(device)  # flatten image to 1'd array
                labels = labels.to(device)

                # step 1: Forward pass: pass each image from batch in the model and compute the neural network from
                # input to output
                outputs = model(images)

                # step2 - compute the objective function (distance between the found to the desirable) and accumulate
                # losses
                loss_func = loss(outputs, labels)  # outputs - found, labels - desirable
                losses.append(loss_func.item())

                if loader is Loaders[0]:  # if data is from test_loader - continue
                    # step 3 - cleaning gradients
                    optimizer.zero_grad()  # zeroes all gradients from previous steps

                    # step 4  - Backward pass: calculate the gradient by passing it through the backward graph of the
                    # neural network
                    loss_func.backward()

                    # step 5 - parameter update based on the current gradient
                    optimizer.step()

                # accuracy calculation:
                total_train += labels.size(0)
                # tensor array of the highest value neuron for each image in the batch
                output_neurons = torch.max(outputs.data, 1)[1]
                # comparison between the output_neurons and the
                # labels.data to accumulate success prediction over epoch
                acc_train += output_neurons.eq(labels.data).sum().item()

                # show progress:
                if loader is Loaders[0] and step == train_samples - 1:
                    # final calculation for loss average and accuracy percentage
                    loss_avg = torch.tensor(losses).mean()
                    accuracy = 100 * acc_train / total_train
                    print("Train: Epoch {}/{}, Samples: {}, Loss: {:.4f}, Accuracy: {:.2f}%".format(epoch + 1, n_epochs,
                                                                                                    train_samples,
                                                                                                    loss_avg,
                                                                                                    accuracy))
                if loader is Loaders[1] and step == test_samples - 1:
                    # final calculation for loss average and accuracy percentage
                    loss_avg = torch.tensor(losses).mean()
                    accuracy = 100 * acc_train / total_train
                    print("Test: Epoch {}/{}, Samples: {}, Loss: {:.4f}, Accuracy: {:.2f}%".format(epoch + 1, n_epochs,
                                                                                                   test_samples,
                                                                                                   loss_avg,
                                                                                                   accuracy))
    # stage screen massage:
    print("Training and testing the model completed\n")
    print("Initiating final test - classify 5 images\n")

    # present result to screen
    (images, labels) = iter(Loaders[1]).next()  # single batch of images from test_loader
    outputs = model(images.reshape(-1, 28 * 28).to(device))  # repeat step 1: Forward pass: Compute Loss
    images = images.numpy()  # convert torch.Tensor to numpy array
    prediction = torch.max(outputs.data, 1)[1]  # take prediction for images of the batch

    for i in range(5):  # final test - show first 5 images and model prediction
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.gray)  # plt.cm.gray for gray scale
        plt.title("Actual: {}\nPrediction: {}".format(
            letter_digit[labels[i]], letter_digit[prediction[i]]),
            fontsize=16, family='serif')
        plt.show()

    # stage screen massage:
    print("Done")


if __name__ == '__main__':
    main()
