import random
from typing import Set

import matplotlib.pyplot as plt
import numpy as np
import tflearn
import tflearn.datasets.cifar10 as cifar
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn import ImageAugmentation

# Hyperparams
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 64

# Augmentation
AUGMENT = True
MAX_ROTATION_ANGLE = 10
MIRROR_IMAGE = True

# Model name (For saving purposes)
MODEL_NAME = 'cifar10_model'

# Set if training or using model
TRAINING = True

# Label explanations
CIFAR_LABELS = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


def mnist_model(lr, tbv=0, activation='relu', aug=None):
    network = input_data(shape=[32, 32, 3], name='input', data_augmentation=aug)

    network = conv_2d(network, 32, 3, activation=activation)
    network = max_pool_2d(network, 2)

    network = conv_2d(network, 64, 2, activation=activation)
    network = max_pool_2d(network, 2)

    network = dropout(network, 0.3)
    network = fully_connected(network, 1024, activation=activation)
    network = dropout(network, 0.7)

    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy')

    return tflearn.DNN(network, tensorboard_verbose=tbv)


def improved_model(lr, tbv=0, activation='relu', aug=None):
    pass


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
              batch_size=batch_size, run_id='cnn_cifar10', snapshot_step=5000, snapshot_epoch=True)
    if save:
        model.save("models/" + mn)


def augmentation():
    aug = ImageAugmentation()

    aug.add_random_rotation(max_angle=MAX_ROTATION_ANGLE)
    if MIRROR_IMAGE:
        aug.add_random_flip_leftright()

    return aug


def plt_random_predictions(model, val_img, val_labels):
    plt.figure()
    for i in range(9):
        ri = random.randint(0, 1000)
        output = np.argmax(model.predict([val_img[ri]]))
        correct = np.argmax(val_labels[ri])

        plt.subplot(3, 3, i + 1)
        plt.imshow(val_img[ri].reshape([32, 32, 3]), cmap='gray_r')
        plt.title("Predicted output: {}\nReal value: {}".format(CIFAR_LABELS[output], CIFAR_LABELS[correct]))
        plt.yticks([])
        plt.xticks([])
    plt.show()


def plt_wrong_predictions(model, val_img, val_labels):
    plt.figure()
    wrongs = []
    j = 0
    print("Acquiring wrong predictions")
    # Check wheter a prediction matches a label, if not, add it to wrong predictions
    while j < len(val_img):
        prediction = model.predict([val_img[j]])
        if np.argmax(prediction) != np.argmax(val_labels[j]):
            wrongs.append(j)
        j += 1

    print("Wrong predictions acquired. {}/{} wrong predictions".format(len(wrongs), len(val_img)))

    # Add the wrong predictions to the plot
    for i in range(9):
        num = wrongs.pop(random.randint(0, len(wrongs) - 1))
        output = np.argmax(model.predict([val_img[num]]))
        correct = np.argmax(val_labels[num])

        plt.subplot(3, 3, i + 1)
        plt.imshow(val_img[num].reshape([32, 32, 3]), cmap='gray_r')
        plt.title("Predicted output: {}\nReal value: {}".format(CIFAR_LABELS[output], CIFAR_LABELS[correct]))
        plt.yticks([])
        plt.xticks([])

    plt.show()


def get_data_information(data, labels, plot=False):
    print("Number of Images:", len(data))
    print("Number of Labels::", len(labels))

    if plot:
        unique, count = np.unique(labels, axis=0, return_counts=True)
        plt.plot()
        plt.bar(range(len(count)), count.tolist(), width=1 / 1.5)
        plt.axes().set_xticklabels([i[0] for i in CIFAR_LABELS])
        plt.show()


def show_random_image(data, labels):
    ri = random.randint(0, len(labels))

    plt.figure()
    plt.title(CIFAR_LABELS[int(np.argmax(labels[ri]))])
    plt.imshow(data[ri])
    plt.yticks([])
    plt.xticks([])
    plt.show()


def main():
    # Get training and validation data and labels
    train, validation = cifar.load_data(one_hot=True)
    train_img, train_labels = train
    validation_img, validation_labels = validation

    # Reshape images
    train_img = train_img.reshape([-1, 32, 32, 3])
    validation_img = validation_img.reshape([-1, 32, 32, 3])

    # get_data_information(train_img, train_labels, plot=False)
    # show_random_image(train_img, train_labels)

    aug = augmentation() if AUGMENT else None
    model = mnist_model(LEARNING_RATE, aug=aug, tbv=0)

    if TRAINING:
        train_model(model, epochs=EPOCHS, batch_size=BATCH_SIZE, data=train_img, labels=train_labels,
                    val_data=validation_img, val_labels=validation_labels, save=True, mn=MODEL_NAME)
    else:
        try:
            print("Loading model:", MODEL_NAME)
            model.load("models/" + MODEL_NAME)
            print("Model loaded")
        except ValueError:
            print("Model not found")
            return

    # Print model accuracy
    print("Model accuracy on test data: {}%".format(100 * model.evaluate(validation_img, validation_labels)[0]))

    # Plot wrong predictions
    plt_random_predictions(model, validation_img, validation_labels)


if __name__ == '__main__':
    main()
