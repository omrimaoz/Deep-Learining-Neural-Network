# imports:
import tensorflow as tf
from tensorflow import keras  # high level API for machine learning. use tensorflow as backend
import matplotlib.pyplot as plt
import numpy as np
import random
from Decompress_Dataset import decompress_zlibFile


def split_dataset(data):
    # define constant ratio for train and test data division
    K = int(data.shape[0] * 0.875)

    # define numpy arrays
    train_images = np.empty(shape=(K, 28, 28))
    train_labels = np.empty(shape=K, dtype=np.int32)
    test_images = np.empty(shape=(data.shape[0] - K, 28, 28))
    test_labels = np.empty(shape=(data.shape[0] - K), dtype=np.int32)

    # shuffle images
    index = [i for i in range(data.shape[0])]
    random.shuffle(index)

    train_index = index[:K]
    test_index = index[K:]

    # fill numpy arrays with images data, define label for each image
    i = 0
    for num in train_index:
        train_images[i] = data[num]
        train_labels[i] = num // 4000  # TODO check if 4000 is true
        i += 1
    i = 0
    for num in test_index:
        test_images[i] = data[num]
        test_labels[i] = num // 4000  # TODO check if 4000 is true
        i += 1

    # stage screen massage:
    print("Successfully prepared data\n")

    return train_images, train_labels, test_images, test_labels


def main():
    # data load - N=104000
    # data = keras.datasets.fashion_mnist
    dataset = decompress_zlibFile("compressed_Dataset")  # dataset is 3d numpy array flatten into 2d array.

    # define train and test data sets
    train_images, train_labels, test_images, test_labels = split_dataset(dataset)

    # model hyper-parameters
    n_epochs = 4
    n_classes = 26  # number of letters - will be the the number of output neurons
    input_layer_neurons = (28, 28)  # 28x28 pixels of image representation
    hidden_layer = 128  # number of neurons in the hidden layer

    # label names
    letter_digit = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
                    ]

    # work with values in the interval [0,1] to represent pixels
    train_images = train_images / 255.0
    test_image = test_images / 255.0

    # define the model - create layers.
    # layer 1 - input layer, layer inputs are the 28x28 pixels of the image flatten
    # into a 1 dimension array.
    # layer 2,3 - a dense layer, that means a fully connected layer with 128 neurons
    # and an activation function - 'relu' - rectified linear unit.
    # layer 4 - a dense layer and the output layer, that means a fully connected layer with
    # 26 neurons and an activation function - 'softmax' - each neuron get a probability value
    # and each neuron represent one of the clothing item define in letter_digit

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_layer_neurons),
        keras.layers.Dense(hidden_layer, activation="relu"),
        keras.layers.Dense(hidden_layer, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax")
    ])

    # optimiser - define optimizer object that will hold the current state and will update the parameters based on
    # the computed gradients.
    # optim.Adam is an optimization algorithms that works very well with this sort of classification
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # stage screen massage:
    print("Successfully defines model\n")
    print("Start training and testing the model")

    # model fitness function
    # epochs - define how many time the model see the train images from the data
    model.fit(train_images, train_labels, epochs=n_epochs)

    # save the model
    # model.save("English_Letters_model.h5") #h5 is the extension in tf and keras

    '''load the model - for that skip code lines: 72 -> 99 and uncomment code line below'''
    # model = keras.models.load_model("English_Letters_model.h5")

    # test model accuracy
    result = model.evaluate(test_image, test_labels)
    print("Model Accuracy is: {}%".format((result[1] * 100)[0:3]))

    # stage screen massage:
    print("Training and testing the model completed\n")
    print("Initiating final test - classify 5 images\n")

    # make a prediction - find it's index
    prediction = model.predict(test_images)
    for i in range(5):
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.gray)  # plt.cm.gray for gray scale
        plt.title("Actual: {}\nPrediction: {}".format(
            letter_digit[test_labels[i]], letter_digit[np.argmax(prediction[i])]),
            fontsize=16, family='serif')
        plt.show()

    # stage screen massage:
    print("Done")


if __name__ == '__main__':
    main()
