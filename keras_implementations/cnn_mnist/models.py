from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.activations import relu, softmax
from keras.optimizers import Adam

import numpy as np


def ann_model(def_activation=relu):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(128, activation=def_activation))
    model.add(Dense(128, activation=def_activation))
    model.add(Dense(10, activation=softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def cnn_model(lr = Adam().lr, def_activation=relu):
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=[28,28,1], activation=def_activation))
    model.add(MaxPool2D(pool_size=2))

    model.add(Conv2D(filters=64, kernel_size=(2,2), activation=def_activation))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(rate=0.3))

    model.add(Dense(1024, activation=def_activation))
    model.add(Dropout(rate=0.7))

    model.add(Dense(10, activation=softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
