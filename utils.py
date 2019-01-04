import matplotlib.pyplot as plt
import random
import numpy as np


def keras_show_random_predictions(data, labels, predictions, shape, wrong=False):
    plt.figure()

    if wrong:
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

    for i in range(9):
        plt.subplot(3, 3, i + 1)

        num = wrongs.pop(random.randint(0, len(wrongs)-1)) if wrong else random.randint(0, len(labels)-1)
        output = np.argmax(predictions[num])
        correct = labels[num]

        show_image_and_label(plt, data[num].reshape(shape), "Predicted output: {}\nReal value: {}".format(output, correct))
    plt.show()


def show_image_and_label(plot, img, title):
    p = plot if plot is not None else plt
    p.xticks([])
    p.yticks([])
    p.imshow(img, cmap='Greys')
    p.title(title)
    return p


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
