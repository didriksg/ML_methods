import time

from keras.datasets import mnist
from keras.utils import normalize

import matplotlib.pyplot as plt
import keras_implementations.cnn_mnist.models as models

import os

# Params
LEARNING_RATE = 0.0001
EPOCHS = 1
BATCH_SIZE = 32

TRAINING = True
SHOW_WRONGS = False
MODEL_NAME = 'mnist_model' + str(int(time.time()))
TBV = 0


def show_random_prediction(model, val_data, val_labels):
    pass


def show_random_wrong_prediction(model, val_data, val_labels):
    pass


def show_image_and_label(img, label):
    plt.figure()
    plt.title(label)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='Greys')
    plt.show()


def train_model(model, train_data, train_labels, batch_size, epochs, save=True, mn="mnist_model", tbv=0):
    # Fit the model according to params
    model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    if save:
        path = "models/"+mn
        if not os.path.exists(path):
            os.makedirs(path)
        model.save(path)


def main():
    # Load data
    # Data are array of np arrays with pixel intensity from 0-255 in one channel, as the data is greyscaled
    # Labels is an array containing the label as an int on the associated data index
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Show an image to verify that the data is loaded and understood correctly
    # show_image_and_label(x_train[0], y_train[0])
    # show_image_and_label(x_test[0], y_test[0])

    # Normalize the data
    x_train = normalize(x_train, axis=1)
    x_test = normalize(x_test, axis=1)

    # Import the model from the models defined in 'models.py'
    model = models.cnn_model()

    # If we are training the model, then train the model
    if TRAINING:
        train_model(model, x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, mn=MODEL_NAME, tbv=TBV)
    else:
        prediction_option = show_random_wrong_prediction if SHOW_WRONGS else show_random_prediction
        prediction_option(model, x_test, y_test)


if __name__ == '__main__':
    main()
