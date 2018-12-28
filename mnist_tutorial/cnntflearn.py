import random

import matplotlib.pyplot as plt
import numpy as np
import tflearn
import tflearn.datasets.mnist as mnist
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression

# Hyperparams
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 128

# Specify whether or not you want to train
TRAINING = True


def simple_model(learning_rate=0.01, tbv=0, activation='relu'):
    network = input_data(shape=[1, 28, 28], name='input')

    network = conv_2d(network, 32, 2, activation=activation)
    network = max_pool_2d(network, 2)

    network = conv_2d(network, 64, 2, activation=activation)
    network = max_pool_2d(network, 2)

    network = fully_connected(network, 1024, activation=activation)
    network = dropout(network, 0.8)

    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

    return tflearn.DNN(network, tensorboard_verbose=tbv)


def advanced_model(learning_rate=0.01, activation='relu', tbv=0):
    adam = tflearn.optimizers.Adam(0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, )

    n = input_data(shape=[1, 28, 28], name='input')

    n = conv_2d(n, 64, 5, activation=activation)
    n = conv_2d(n, 128, 5, activation=activation)
    n = max_pool_2d(n, 2, strides=2)

    n = conv_2d(n, 256, 5, activation=activation)
    n = conv_2d(n, 256, 3, activation=activation)
    n = max_pool_2d(n, 2)
    n = dropout(n, 0.2)

    n = conv_2d(n, 512, 3, activation=activation)
    n = dropout(n, 0.2)

    n = conv_2d(n, 512, 3, activation=activation)
    n = max_pool_2d(n, 2)

    n = flatten(n)
    n = dropout(n, 0.5)
    n = fully_connected(n, 2048, activation=activation)

    n = fully_connected(n, 10, activation='softmax')

    n = regression(n, optimizer=adam, learning_rate=learning_rate, loss='categorical_crossentropy')
    return tflearn.DNN(n, tensorboard_verbose=tbv)


def train_model(model, epochs, batch_size, image, labels, validation_image, validation_labels, save_model=True,
                model_name="tflearnmodel"):
    model.fit(image, labels, n_epoch=epochs, validation_set=(validation_image, validation_labels), show_metric=True,
              batch_size=batch_size, run_id='mnist', snapshot_step=50000, snapshot_epoch=False)
    if save_model:
        model.save(model_name)


def plt_prediction(model, test_data, labels):
    plt.figure()
    for i in range(9):
        ri = random.randint(0, 1000)
        output = model.predict([test_data[ri]])

        plt.subplot(3, 3, i + 1)
        plt.imshow(test_data[ri].reshape([28, 28]), cmap='gray_r')
        plt.title("Predicted output: {}\nReal value: {}".format(np.argmax(output), np.argmax(labels[ri])))
    plt.show()


def get_model(type, tbv=0):
    model = advanced_model(LEARNING_RATE, tbv) if type == 'advanced' else simple_model(LEARNING_RATE, tbv)
    if TRAINING:
        return model

    else:
        try:
            model.load(type)
            print("Model loaded")
            return model
        except ValueError:
            print("Model not found. Check that type is specified correct, or traing network")
            return None


def main():
    # Specify model type. (simple or advanced)
    model_type = 'simple'

    # Load data to training and validation data
    X, Y, test_x, test_y = mnist.load_data(one_hot=True)

    # Reshape to fit into input
    X = X.reshape([-1, 1, 28, 28])
    test_x = test_x.reshape([-1, 1, 28, 28])

    # Get model
    model = get_model(model_type)

    if TRAINING:
        train_model(model, EPOCHS, BATCH_SIZE, X, Y, test_x, test_y, model_name=model_type)

    # Print model accuracy
    print("Model accuracy on test data: {}".format(model.evaluate(test_x, test_y)))

    # Plot predictions on random images from the testdata
    plt_prediction(model, test_x, test_y)


if __name__ == '__main__':
    main()
