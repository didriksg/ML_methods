import time
import os

from keras.datasets import mnist
from keras.utils import normalize

import matplotlib.pyplot as plt

import keras_implementations.cnn_mnist.models as models

# Hyperparams
LEARNING_RATE = 0.0001
EPOCHS = 5
BATCH_SIZE = 32

# Settings
TRAINING = True
SAVE_MODEL = True
SHOW_WRONGS = False
IS_CNN = True
VERBOSE = 1

# Constants
MODELS_BASE_DIR = "models/"
MODEL_NAME = 'mnist_model'


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


def train_model(model, train_data, train_labels, val_data, batch_size, epochs, save=True, tbv=0):
    # Fit the model according to params
    model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=tbv,
              validation_data=val_data)

    if save:
        model.save_weights(MODELS_BASE_DIR + MODEL_NAME + ".h5")


def main():
    # Load data
    # Data are array of np arrays with pixel intensity from 0-255 in one channel, as the data is greyscaled
    # Labels is an array containing the label as an int on the associated data index
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Show an image to verify that the data is loaded and understood correctly
    # show_image_and_label(x_train[0], y_train[0])
    # show_image_and_label(x_test[0], y_test[0])

    # Normalize the data
    if IS_CNN:
        x_train = x_train.reshape([-1,28,28,1])
        x_test = x_test.reshape([-1,28,28,1])
    else:
        x_train = normalize(x_train)
        x_test = normalize(x_test)

    # Import the model from the models defined in 'models.py'
    model = models.cnn_model(lr=LEARNING_RATE) if IS_CNN else models.ann_model(lr=LEARNING_RATE)
    # If we are training the model, then train the model
    if TRAINING:
        train_model(model, x_train, y_train, (x_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS, tbv=VERBOSE,
                    save=SAVE_MODEL)
    # If not, do a prediction; Either a random, or only show the wrongly predicted ones
    else:
        prediction_option = show_random_wrong_prediction if SHOW_WRONGS else show_random_prediction
        prediction_option(model, x_test, y_test)


if __name__ == '__main__':
    main()
