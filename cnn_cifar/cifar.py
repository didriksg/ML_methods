import random

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



def main():
    # Set model name
    model_name = MODEL_NAME

    # Get training and validation data and labels
    train_data, train_labels, validation_data, validation_labels = cifar.load_data(one_hot=True)

    plt.figure()
    test_image = train_data[0]

    plt.figure()
    plt.imshow(test_image)


if __name__ == '__main__':
    main()