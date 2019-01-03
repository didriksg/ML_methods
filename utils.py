import matplotlib.pyplot as plt
import random
import numpy as np


def keras_show_random_predictions(data, labels, predictions, shape):
    """
    Keras version: Plot random predictions. Can be both wrong and correct

    :param data: Test data
    :param labels: Test labels
    :param predictions: Predictions done on a given model
    :return:
    """

    plt.figure()
    for i in range(9):
        ri = random.randint(0, 1000)
        output = np.argmax(predictions[ri])
        correct = labels[ri]

        plt.subplot(3, 3, i + 1)
        plt.imshow(data[ri].reshape(shape), cmap='Greys')
        plt.title("Predicted output: {}\nReal value: {}".format(output, correct))
    plt.show()


def keras_show_wrong_predictions(data, labels, predictions, shape):
    """
    Keras version: Plot random wrong predictions and print number of wrong predictions

    :param data: Test data
    :param labels: Test labels
    :param predictions: Predictions done on a given model
    :return:
    """

    plt.figure()
    wrongs = []
    j = 0
    print("Acquiring wrong predictions")
    # Check whether a prediction matches a label, if not, add it to wrong predictions
    while j < len(data):
        prediction = predictions[j]
        if np.argmax(prediction) != labels[j]:
            wrongs.append(j)
        j += 1

    print("Wrong predictions acquired. {}/{} wrong predictions".format(len(wrongs), len(data)))

    # Add the wrong predictions to the plot
    for i in range(9):
        num = wrongs.pop(random.randint(0, len(wrongs) - 1))
        output = np.argmax(predictions[num])
        correct = labels[num]

        plt.subplot(3, 3, i + 1)
        plt.imshow(data[num].reshape(shape), cmap='Greys')
        plt.title("Predicted output: {}\nReal value: {}".format(output, correct))
    plt.show()


def show_image_and_label(img, label):
    plt.figure()
    plt.title(label)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='Greys')
    plt.show()

def evaluate_model(model, val_data, val_labels):
    print(divider(70))
    print("Evaluating model")
    res = model.evaluate(val_data, val_labels)
    print("Done evaluating")
    print(divider(70))
    print("Model results:")
    print("Model validation loss: {:.5}\nModel Accuracy: {:.2%}".format(res[0], res[1]))
    print(divider(70))



def predict(model, data):
    if not isinstance(data, list):
        data = [data]
    return model.predict(data)


def divider(length):
    s = ""
    for _ in range(length):
        s += str("-")
    return s