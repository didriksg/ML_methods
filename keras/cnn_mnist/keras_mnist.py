from keras.datasets import mnist

import matplotlib.pyplot as plt
import numpy as np


# Params
LEARNING_RATE = 0.0001
EPOCHS = 50


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


def main():
    # Load data
    # Data are array of np arrays with pixel intensity from 0-255 in one channel, as the data is greyscaled
    # Labels is an array containing the label as an int on the associated data index
    (train_data, train_labels), (val_data, val_labels) = mnist.load_data()

    # Show an image to verify that the data is loaded and understood correctly
    show_image_and_label(train_data[0], train_labels[0])
    show_image_and_label(val_data[0], val_labels[0])




if __name__ == '__main__':
    main()
