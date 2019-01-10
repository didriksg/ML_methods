import random, time, glob
import matplotlib.pyplot as plt
import cv2
import numpy as np

from constants import NAME_WITH_TIME


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

    numbs = [i for i in range(len(labels))]
    for i in range(min(9,len(labels))):
        plt.subplot(3, 3, i + 1)
        random.seed()
        num = wrongs.pop(random.randint(0, len(wrongs) - 1)) if wrong else numbs.pop(random.randint(0, len(numbs) - 1))
        output = np.argmax(predictions[num])
        correct = labels[num]

        if len(labels) > 1:
            show_image_and_label(plt, data[num].reshape(shape),
                             "Predicted output: {}\nReal value: {}".format(output, correct))
    plt.show()


def show_image_and_label(plot, img, title):
    p = plot if plot is not None else plt
    p.xticks([])
    p.yticks([])
    p.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
    return res


def predict(model, data):
    if not isinstance(data, list):
        data = [data]
    return model.predict(data)


def divider(length):
    s = ""
    for _ in range(length):
        s += str("-")
    return s


def name_model(model_name, augmented):
    return model_name + ("_augmented" if augmented else "") + ("_{}".format(int(time.time())) if NAME_WITH_TIME else "")
