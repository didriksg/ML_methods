from keras.datasets import mnist
from keras.utils import normalize

import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '../../'))

from utils import keras_show_random_predictions as show_predictions, show_image_and_label
from utils import evaluate_model, predict
from keras_implementations.cnn_mnist.models import cnn_model, fc_model

# Hyperparams
LEARNING_RATE = 0.0001
EPOCHS = 5
BATCH_SIZE = 32

# Settings
TRAINING = False
SAVE_MODEL = True
USE_CNN = True
SHOW_WRONGS = True

VERBOSE = 1

# Constants
MODELS_BASE_DIR = "weights/"
MODEL_NAME = 'mnist_model' + ('_cnn' if USE_CNN else '_full')
EXTENSION = '.h5'


def train_model(model, train_data, train_labels, val_data, batch_size, epochs, save=True, tbv=0):
    # Fit the model according to params
    model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, verbose=tbv,
              validation_data=val_data)

    if save:
        model.save_weights(MODELS_BASE_DIR + MODEL_NAME + EXTENSION)


def main():
    # Load data
    # Data are array of np arrays with pixel intensity from 0-255 in one channel, as the data is greyscale
    # Labels is an array containing the label as an int on the associated data index
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Show an image to verify that the data is loaded and understood correctly
    # show_image_and_label(x_train[0], y_train[0])
    # show_image_and_label(x_test[0], y_test[0])

    # Preprocess data to fit their network
    if USE_CNN:
        x_train = x_train.reshape([-1, 28, 28, 1])
        x_test = x_test.reshape([-1, 28, 28, 1])
    else:
        x_train = normalize(x_train)
        x_test = normalize(x_test)

    # Import the model from the models defined in 'models.py'
    model = cnn_model(lr=LEARNING_RATE, shape=x_train.shape[1:]) if USE_CNN else fc_model(lr=LEARNING_RATE)
    model.fit(augment=True)
    # If we are training the model, then train the model
    if TRAINING:
        train_model(model, x_train, y_train, (x_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS, tbv=VERBOSE,
                    save=SAVE_MODEL)
    # If not, do a prediction; Either a random, or only show the wrongly predicted ones
    else:
        print("Loading model:", MODEL_NAME)
        model.load_weights(MODELS_BASE_DIR + MODEL_NAME + EXTENSION)
        print("Model loaded")

        evaluate_model(model, x_test, y_test)
    prediction = predict(model, x_test)

    show_predictions(x_test, y_test, prediction, [28, 28], wrong=SHOW_WRONGS)

if __name__ == '__main__':
    main()
