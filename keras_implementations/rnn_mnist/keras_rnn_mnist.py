import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM, LSTM
from keras.datasets import mnist, cifar10
from keras.optimizers import Adam

from training import train_model

LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 10

IMAGE_WIDTH = IMAGE_HEIGHT = 28
CHANNELS = 1
MODEL_NAME = "rnn_test"


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape([-1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS])
    x_test = x_test.reshape([-1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS])

    mod = model(LEARNING_RATE, x_train.shape[1:])
    train_model(mod, x_train, y_train, (x_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS, augment=None,
                model_name=MODEL_NAME)


def model(lr, shape):
    model = Sequential()

    model.add(CuDNNLSTM(128, input_shape=shape, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(CuDNNLSTM(128))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=Adam(lr=lr, decay=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    main()
