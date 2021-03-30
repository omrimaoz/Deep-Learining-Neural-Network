import tensorflow as tf
from tensorflow import keras  # high level API for machine learning. use tensorflow as backend
import matplotlib.pyplot as plt
import numpy as np
import random


def show_image(image):
    plt.imshow(image, cmap=plt.cm.binary)  # plt.cm.binary for gray scale
    plt.show()


def split_dataset(data):
    K = int(data.shape[0] * 0.6)
    M = 100
    train_images = np.empty(shape=(K * M, 30, 40))
    train_labels = np.empty(shape=(K * M), dtype=np.int32)
    test_images = np.empty(shape=(data.shape[0] - K, 30, 40))
    test_labels = np.empty(shape=(data.shape[0] - K), dtype=np.int32)

    index = [i for i in range(data.shape[0])]
    random.shuffle(index)

    train_index = index[:K]
    test_index = index[K:]
    for j in range(M):
        i = 0
        for num in train_index:
            train_images[K * j + i] = data[num]
            train_labels[K * j + i] = num // 55  # TODO check if 55 is true
            i += 1
    i = 0
    for num in test_index:
        test_images[i] = data[num]
        test_labels[i] = num // 55  # TODO check if 55 is true
        i += 1
    return train_images, train_labels, test_images, test_labels


def main():
    # data load - N=104000
    data = keras.datasets.fashion_mnist
    data = np.loadtxt("Handwrite_Dataset.txt")  # dataset is 3d numpy array flatten into 2d array.
    images_d = 28  # dataset images width

    # reshape the data into 3d numpy array - shape=(3640,30,40)
    # 3640 images constructed the dataset
    # 30x40 is the image ratio
    reshaped_data = data.reshape(
        data.shape[0], data.shape[1] // images_d, images_d)
    reshaped_data = reshaped_data[550:1980]
    print(reshaped_data.shape)
    # define train and test data sets
    train_images, train_labels, test_images, test_labels = split_dataset(reshaped_data)
    print(train_images[0], train_labels[0])
    # label names
    # letter_digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    #                 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    #                 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    #                 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    #                 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    #                 ]
    letter_digit = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
                    ]

    # work with values in the interval [0,1] to represent pixels
    train_images = train_images / 255.0
    test_image = test_images / 255.0

    # define the model - create layers.
    # layer 1 - input layer, layer inputs are the 28x28 pixels of the image flatten
    # into a 1 dimension numpy array.
    # layer 2 - a dense layer, that means a fully connected layer with 128 neurons
    # and an activation function - 'relu' - rectified linear unit.
    # layer 3 - a dense layer and the output layer, that means a fully connected layer with
    # 10 neurons and an activation function - 'softmax' - each neuron get a probability value
    # and each neuron represent one of the clothing item define in letter_digit

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(26, activation="softmax")
    ])

    # model parameters
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # model fitness function
    # epochs - define how many time the model see the train images from the data
    model.fit(train_images, train_labels, epochs=5)

    # test model accuracy
    result = model.evaluate(test_image, test_labels)
    print("Model Accuracy is: " + str(result[1] * 100)[0:3] + "%")

    # make a prediction - find it's index
    prediction = model.predict(test_images)
    for i in range(5):
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)  # plt.cm.binary for gray scale
        plt.xlabel("Actual: " + letter_digit[test_labels[i]])
        plt.title("Prediction: " + letter_digit[np.argmax(prediction[i])])
        plt.show()

    # save the model
    model.save("text_classification_model.h5") #h5 is the extension in tf and keras


if __name__ == '__main__':
    main()
