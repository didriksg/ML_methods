import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

import matplotlib.pyplot as plt
import numpy as np
import random

LEARNING_RATE = 0.01
EPOCHS = 10


def simple_network(learning_rate=0.01):
    cn = input_data(shape=[None, 28, 28, 1], name='input')

    cn = conv_2d(cn, 32, 2, activation='relu')
    cn = max_pool_2d(cn, 2)

    cn = conv_2d(cn, 64, 2, activation='relu')
    cn = max_pool_2d(cn, 2)

    cn = fully_connected(cn, 1024, activation='relu')
    cn = dropout(cn, 0.8)

    cn = fully_connected(cn, 10, activation='softmax')
    cn = regression(cn, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

    return tflearn.DNN(cn)


def train_model(model, X, Y, epochs, test_x, test_y, save_model=True, model_name="tflearnmodel"):
    model.fit(X, Y, n_epoch=epochs, validation_set=(test_x, test_y), snapshot_step=500, show_metric=True,
              run_id='mnist')
    if save_model:
        model.save(model_name)


def load_model(model, model_name='tflearnmodel'):
    return model.load(model_name)



def predict_from_model(model, data):
    return model.predict(data.reshape([28, 28]))


def plt_result(model, test_data):
    plt.figure()
    for i in range(9):
        ri = random.randint(0, 1000)
        output = model.predict([test_data[ri]])

        plt.subplot(3, 3, i + 1)
        plt.imshow(test_data[ri].reshape([28, 28]), cmap='gray_r')
        plt.title("Predicted output: {}".format(np.argmax(output)))
    plt.show()


def main():
    X, Y, test_x, test_y = mnist.load_data(one_hot=True)

    X = X.reshape([-1, 28, 28, 1])
    test_x = test_x.reshape([-1, 28, 28, 1])

    model = simple_network(LEARNING_RATE)
    plt_result(model, test_x)

    # train_model(model, X, Y, EPOCHS, test_x, test_y)
    load_model(model, 'tflearnmodel')
    plt_result(model, test_x)


if __name__ == '__main___':
    main()