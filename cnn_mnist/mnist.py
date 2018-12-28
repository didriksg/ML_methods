import random

import matplotlib.pyplot as plt
import numpy as np
import tflearn
import tflearn.datasets.mnist as mnist
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Hyperparams
LEARNING_RATE = 0.005
EPOCHS = 100
BATCH_SIZE = 64

MODEL_NAME = "simple"

# Augmentation settings:
AUGMENT = True
MAX_ROT_DEG = 20

# Specify whether or not you want to train
TRAINING = False


def get_model(learning_rate=0.01, tbv=0, activation='relu', aug=None):
    """
    Get the model used for training and predictions
    :param learning_rate: Learning rate for the model
    :param tbv: Tensorboard verbose
    :param activation: Activation function used on all layers (except last, where softmax is used)
    :param aug: Data augmentation
    :return: The model
    """
    # Input(28,28,1) -> conv2d(32,3), max2d(2) -> conv2d(64,2) -> max2d(2) -> dropout(0.3) -> Dense(1024) ->
    # drop(0.7) -> Dense(10, Softmax)

    network = input_data(shape=[28, 28, 1], name='input', data_augmentation=aug)

    network = conv_2d(network, 32, 3, activation=activation)
    network = max_pool_2d(network, 2)

    network = conv_2d(network, 64, 2, activation=activation)
    network = max_pool_2d(network, 2)

    network = dropout(network, 0.3)
    network = fully_connected(network, 1024, activation=activation)
    network = dropout(network, 0.7)

    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

    return tflearn.DNN(network, tensorboard_verbose=tbv)


def get_augmentation(max_rot_deg=20):
    """
    Augmentations used to prevent overfitting
    :param max_rot_deg: Maximum rotation degree
    :return: The augmentation
    """
    aug = tflearn.ImageAugmentation()
    aug.add_random_rotation(max_rot_deg)

    return aug


def train_model(model, epochs, batch_size, data, labels, val_data, val_labels, save=True, mn="model"):
    """
    Train the given model with given params
    :param model: The model to be trained
    :param epochs: Number of epochs to train the model for
    :param batch_size: Batch size used
    :param data: Training data
    :param labels: Training labels
    :param val_data: Validation data
    :param val_labels: Validation labels
    :param save: Set to 'True' to save the model
    :param mn: Model name
    :return:
    """

    model.fit(data, labels, n_epoch=epochs, validation_set=(val_data, val_labels), show_metric=True,
              batch_size=batch_size, run_id='cnn_mnist', snapshot_step=5000, snapshot_epoch=True)
    if save:
        model.save("models/" + mn)


def plt_random_predictions(model, data, labels):
    """
    Plot random predictions. Can be both wrong and correct
    :param model: Trained model
    :param data: Test data
    :param labels: Test labels
    :return:
    """

    plt.figure()
    for i in range(9):
        ri = random.randint(0, 1000)
        output = model.predict([data[ri]])

        plt.subplot(3, 3, i + 1)
        plt.imshow(data[ri].reshape([28, 28]), cmap='gray_r')
        plt.title("Predicted output: {}\nReal value: {}".format(np.argmax(output), np.argmax(labels[ri])))
    plt.show()


def plt_random_wrong_predictions(model, data, labels, n=9):
    """
    Plot random wrong predictions and print number of wrong predictions
    :param model: Trained model
    :param data: Test data
    :param labels: Test labels
    :param n: Number of predictions shown
    :return:
    """

    plt.figure()
    wrongs = []
    j = 0
    print("Acquiring wrong predictions")
    # Check wheter a prediction matches a label, if not, add it to wrong predictions
    while j < len(data):
        prediction = model.predict([data[j]])
        if np.argmax(prediction) != np.argmax(labels[j]):
            wrongs.append(j)
        j += 1

    print("Wrong predictions acquired. {}/{} wrong predictions".format(len(wrongs), len(data)))

    # Add the wrong predictions to the plot
    for i in range(n):
        num = wrongs.pop(random.randint(0, len(wrongs) - 1))
        output = np.argmax(model.predict([data[num]]))
        correct = np.argmax(labels[num])

        plt.subplot(3, 3, i + 1)
        plt.imshow(data[num].reshape([28, 28]), cmap='gray_r')
        plt.title("Predicted output: {}\nReal value: {}".format(output, correct))

    plt.show()


def main():
    # Specify model name. (for saving purposes)
    model_name = MODEL_NAME

    # Load data to training and validation data
    train_data, train_labels, test_data, test_labels = mnist.load_data(one_hot=True)

    # Reshape to fit into input
    train_data = train_data.reshape([-1, 28, 28, 1])
    test_data = test_data.reshape([-1, 28, 28, 1])

    # Setup augmentation if requested:
    aug = get_augmentation(max_rot_deg=MAX_ROT_DEG) if AUGMENT else None

    # Get model
    model = get_model(learning_rate=LEARNING_RATE, tbv=0, activation='relu', aug=aug)

    if not TRAINING:
        try:
            print("Loading model:", model_name)
            model.load("models/" + model_name)
            print("Model loaded")
        except ValueError:
            print("Model not found.")
            return
    else:
        print("Training model:", model_name)
        train_model(model, EPOCHS, BATCH_SIZE, train_data, train_labels, test_data, test_labels, mn=model_name)

    # Print model accuracy
    print("Model accuracy on test data: {}%".format(100 * model.evaluate(test_data, test_labels)[0]))

    # Plot wrong predictions
    plt_random_wrong_predictions(model, test_data, test_labels)


if __name__ == '__main__':
    main()
