from keras.datasets import mnist

import matplotlib.pyplot as plt
import numpy as np

# Params
DATA_PATH = 'data/'
LEARNING_RATE = 0.0001
EPOCHS = 50


def show_random_prediction(model, val_data, val_labels):
    pass


def show_random_wrong_prediction(model, val_data, val_labels):
    pass


def show_image_and_label(img, label):
    plt.figure()
    plt.title(np.argmax(label))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)


def main():
    # Load data
    (train_data, train_labels), (val_data, val_labels) = mnist.load_data(DATA_PATH)

    # Show an image to verify that we loaded the data correctly
    show_image_and_label(train_data[0], train_labels[0])


if __name__ == '__main__':
    main()
