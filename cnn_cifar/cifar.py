import random
from typing import Set

import matplotlib.pyplot as plt
import numpy as np
import tflearn
import tflearn.datasets.cifar10 as cifar
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Hyperparams
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 32

MODEL_NAME = 'cifar_model'

TRAINING = False

CIFAR_LABELS = {'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'}


def cifar_model():
    pass


def augmentation():
    pass


def plt_random():
    pass


def plt_wrong():
    pass


def get_data_information(data, labels, plot=False):
    print("Number of Images:", len(data))
    print("Number of Labels::", len(labels))

    if plot:
        fig, ax = plt.subplot()
        plt.bar(np.arrange(1, 10), [len(i) for i in data])
        ax.set_xlabels(CIFAR_LABELS)
        plt.show()



def main():
    # Set model name
    model_name = MODEL_NAME

    # Get training and validation data and labels
    train, validation = cifar.load_data(one_hot=True)
    train_img, train_labels = train
    validation_img, validation_labels = validation

    get_data_information(train_img, train_labels, plot=True)


if __name__ == '__main__':
    main()
