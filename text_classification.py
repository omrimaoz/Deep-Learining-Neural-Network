import tensorflow as tf
from tensorflow import keras  # high level API for machine learning. use tensorflow as backend
import matplotlib.pyplot as plt
import numpy as np

# decode method using the decoding function (reverse dict)
def decode_review(reverse_word_index, text):
    st = ""
    for i in text:
        if i == 10:
            continue  # skip html keyword "<br>"
        else:
            st = st + " " + reverse_word_index.get(i, "?")
    return st[1:]

def main():
    data = keras.datasets.imdb

    # define train and test data sets
    # word coded into integers
    (train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=880000)

    # word indexes
    word_index = data.get_word_index()

    # define code integer for words that cant be classified to current imdb indexes
    word_index = {k: (v + 3) for k, v in word_index.items()}  # define new encode dict
    word_index["<PAD>"] = 0  # the empty word in order to fill short review to our model length
    word_index["<START>"] = 1  # review beginning - const word
    word_index["<UNK>"] = 2  # Unknown word
    word_index["<UNUSED>"] = 3  # Unused word

    # swap keys and values in word_index dict - the decoding function
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    mapp = {0: "Good Review", 1: "Bad Review"}

    # each review is in different length. our model need constant length review so we will cut long review and add
    # <PAD>s to short reviews
    train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                            maxlen=250)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post",
                                                           maxlen=250)

    # define the model - create layers.
    # layer 1 - takes 10000 words and create vectors in a 2'd axes system by
    # various calculation in order to determine close relation between words.
    # layer 2 =
    # layer 3 - a dense layer, that means a fully connected layer with 16 neurons and
    # an activation function - 'relu' - rectified linear unit.
    # layer 4 - a dense layer and the output layer, that means a fully connected
    # layer with 1 neuron and an activation function - 'softmax' - the neuron get a probability value -
    # between 0 and 1 and that define if the review is good or bad

    model = keras.Sequential()
    model.add(keras.layers.Embedding(880000, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.summary()  # TODO: look what it does

    # model parameters
    # binary_crossentropy - binary = {0,1}, a function to determent the gap between output neuron value from 0 or 1
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # partition train data and labels for validation by the first 10000 reviews
    x_val = train_data[:10000]
    x_train = train_data[10000:]

    y_val = train_labels[:10000]
    y_train = train_labels[10000:]

    # model fitness function
    # epochs - define how many time the model see the train reviews(words) from the data
    # batch_size - define number of reviews to feed the model on battle-neck parts TODO valid it
    # validation_data - TODO
    # verbose - TODO
    model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

    # test model accuracy
    result = model.evaluate(test_data, test_labels)
    print("Model Accuracy is: " + str(result[1] * 100)[0:4] + "%")

    # make a prediction - test if the model can predict the first review is good or bad
    prediction = model.predict(test_data[0])
    print("Review: " + decode_review(reverse_word_index, test_data[0]))
    print("Prediction: " + mapp[int(np.round(prediction[0][0], 0))])
    print("Actual: " + mapp[test_labels[0]])

    # save the model
    # model.save("text_classification_model.h5") #h5 is the extension in tf and keras

    model= keras.model.load_model("text_classification_model.h5")



if __name__ == '__main__':
    main()
