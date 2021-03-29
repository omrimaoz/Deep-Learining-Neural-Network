import tensorflow as tf
from tensorflow import keras  # high level API for machine learning. use tensorflow as backend
import matplotlib.pyplot as plt
import numpy as np


def show_image(image):
    plt.imshow(image, cmap=plt.cm.binary)  # plt.cm.binary for gray scale
    plt.show()


def main():
    data = keras.datasets.fashion_mnist

    # define train and test data sets
    (train_images, train_labels), (test_images, test_labels) = data.load_data()
    print(train_images[0])
    # label names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

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
    # and each neuron represent one of the clothing item define in class_names

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
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
        plt.xlabel("Actual: " + class_names[test_labels[i]])
        plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
        plt.show()

    # save the model
    model.save("text_classification_model.h5") #h5 is the extension in tf and keras


if __name__ == '__main__':
    main()
