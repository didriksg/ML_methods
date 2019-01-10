from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from utils import keras_show_random_predictions
from sklearn.utils import class_weight

import glob
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

from training import train_model_augment

BATCH_SIZE = 1
EPOCHS = 10
MODEL_NAME = "limited_unbalanced_data"

string_to_int_value = {"berry": 0, "bird": 1, "flower": 2, "dog": 3}
reverse = inv_map = {v: k for k, v in string_to_int_value.items()}


def import_data():
    data = []
    labels = []
    for folder in glob.glob('data/*'):
        label = folder.replace("data\\", "")
        label = string_to_int_value[label]

        for file in glob.glob(f"{folder}/*.jpg"):
            im = cv2.imread(file)
            data.append(im)
            labels.append(label)

    return np.array(data), np.array(labels)


def preprocess_data(data, labels):
    unique, indicies, counts = np.unique(labels, return_counts=True, return_index=True)
    print(unique, counts, indicies)

    x_val = []
    y_val = []
    seen = []
    for i in range(len(unique)):
        if labels[i] not in seen:
            x_val.append(data[i])
            y_val.append(labels[i])

    # Shuffle data to have a random order. Seed is set to make to get the same distribution during multiple
    # trainings when testing
    tot = list(zip(data, labels))
    random.seed(123)
    random.shuffle(tot)

    # Distribute into training and testing data
    five_pct = int(0.05 * len(X))
    x_val = X[-five_pct:]
    y_val = Y[-five_pct:]

    new_data, new_labels = zip(*tot)
    return np.array(new_data), np.array(new_labels)


def get_model(lr, shape):
    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal',
               input_shape=shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.20))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer=Adam(lr=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def main():
    # Load the data
    data, labels = import_data()

    # Preprocess data (Effectively only shuffeling at this point
    (x_train, y_train), (x_val, y_val) = preprocess_data(data, labels)

    # Load the model
    model = get_model(0.001, [32, 32, 3])

    # Set the augmentation used. More can possibly be added
    augment = ImageDataGenerator(rotation_range=15, horizontal_flip=True, height_shift_range=2, width_shift_range=2,
                                 zoom_range=0.1)

    # Set the class weights based on the training labels
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

    # Train the model
    train_model_augment(model, train_data=x_train, train_labels=y_train, val_data=x_val, val_labels=y_val,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS, model_name=MODEL_NAME, augment=augment, augment_batch_size=32,
                        augment_validation=False, class_weights=class_weights)

    # model.load_weights(f'weights/beb.h5')
    # keras_show_random_predictions(x_val, y_val, predictions=model.predict(x_val), shape=[32, 32, 3])


if __name__ == '__main__':
    main()
