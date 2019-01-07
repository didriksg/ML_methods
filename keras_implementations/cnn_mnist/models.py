from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam


def improved_cnn(lr, shape, num_classes, def_activation='relu'):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=shape, activation=def_activation))
    model.add(Conv2D(32, (3, 3), activation=def_activation))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation=def_activation))
    model.add(Conv2D(64, (3, 3), activation=def_activation))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation=def_activation))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(lr=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def cnn_model(lr=Adam().lr, shape=None, def_activation='relu'):
    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(3, 3), activation=def_activation, kernel_initializer='he_normal', input_shape=shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation=def_activation, kernel_initializer='he_normal'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.20))

    model.add(Conv2D(64, (3, 3), activation=def_activation, padding='same', kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation=def_activation, padding='same', kernel_initializer='he_normal'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation=def_activation, padding='same', kernel_initializer='he_normal'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation=def_activation))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Adam(lr=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
