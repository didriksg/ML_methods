from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '../../'))
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from training import train_model
from constants import WEIGHTS_BASE_DIR, WEIGHTS_EXTENSION
from cnn_mnist.models import cnn_model, improved_cnn

from utils import keras_show_random_predictions as show_predictions
from utils import evaluate_model, predict, name_model

# Hyperparams
LEARNING_RATE = 0.0005
EPOCHS = 10
BATCH_SIZE = 64
IMAGE_WIDTH = IMAGE_HEIGHT = 28
CHANNELS = 1

# Settings
TRAINING = False
AUGMENT = True
USE_IMPROVED = True
SHOW_PREDICTIONS = True
ONLY_WRONGS = False

# Augmentation settings
ROTATION = 12
WIDTH_SHIFT = 0
HEIGHT_SHIFT = 0
SHEAR_RANGE = 0.0
ZOOM_RANGE = 0.0

VERBOSE = 1

# Constants
MODELS_BASE_DIR = "weights/"
MODEL_NAME = 'fashion_mnist' + ('_improved' if USE_IMPROVED else '')


def main():
    model_name = name_model(MODEL_NAME, AUGMENT)
    # Load data
    # Data are array of np arrays with pixel intensity from 0-255 in one channel, as the data is greyscale
    # Labels is an array containing the label as an int on the associated data index
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Preprocess data to fit their network
    x_train = x_train.reshape([-1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS])
    x_test = x_test.reshape([-1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS])

    # Import the preferred from the models defined in 'models.py'
    if USE_IMPROVED:
        model = improved_cnn(lr=LEARNING_RATE, shape=x_train.shape[1:], num_classes=10)
    else:
        model = cnn_model(lr=LEARNING_RATE, shape=x_train.shape[1:])

    augment = None if not AUGMENT else ImageDataGenerator(rotation_range=ROTATION,
                                                          width_shift_range=WIDTH_SHIFT,
                                                          height_shift_range=HEIGHT_SHIFT,
                                                          shear_range=SHEAR_RANGE,
                                                          zoom_range=ZOOM_RANGE)


    # If we are training the model, then train the model
    if TRAINING:
        train_model(model, x_train, y_train, x_test, y_test,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    augment=augment,
                    model_name=model_name,
                    verbose=VERBOSE)
    # If not, do a prediction; Either a random, or only show the wrongly predicted ones
    else:
        print("Loading model:", model_name)
        if os.path.exists(WEIGHTS_BASE_DIR + model_name + WEIGHTS_EXTENSION):
            model.load_weights(WEIGHTS_BASE_DIR + model_name + WEIGHTS_EXTENSION)
            print("Model loaded")
        else:
            raise FileNotFoundError(f"Weights with filename '{model_name}' was not found")

        evaluate_model(model, x_test, y_test)

    if SHOW_PREDICTIONS:
        show_predictions(x_test, y_test, predict(model, x_test), x_train.shape[1:-1], wrong=ONLY_WRONGS)


if __name__ == '__main__':
    main()
