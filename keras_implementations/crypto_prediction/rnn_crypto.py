# Based on pythonprogramming's tutorial on RNN's

import pandas as pd
from sklearn import preprocessing
from collections import deque
import numpy as np
import random
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM, BatchNormalization
from keras.optimizers import Adam

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from training import train_model

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "BTC-USD"

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.1
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


def classify(current, future):
    return int(float(future) > float(current))


def preprocess_df(df):
    df = df.drop("future", 1)

    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells

    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y


def import_data():
    main_df = pd.DataFrame()

    ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]
    for ratio in ratios:
        dataset = f"crypto_data/{ratio}.csv"

        df = pd.read_csv(dataset, names=["time", "low", "high", "open", "close", "volume"])
        df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

        df.set_index("time", inplace=True)

        df = df[[f"{ratio}_close", f"{ratio}_volume"]]

        if len(main_df) == 0:
            main_df = df
        else:
            main_df = main_df.join(df)

    main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
    main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))

    return main_df


def get_model(lr, shape):
    model = Sequential()
    model.add(CuDNNLSTM(128, input_shape=shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(CuDNNLSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(CuDNNLSTM(128))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=lr, decay=1e-6), metrics=['accuracy'])
    return model


def main():
    main_df = import_data()

    times = sorted(main_df.index.values)
    last_5pct = times[-int(0.05 * len(times))]

    validation_main_df = main_df[(main_df.index >= last_5pct)]
    main_df = main_df[(main_df.index < last_5pct)]

    train_x, train_y = preprocess_df(main_df)
    val_x, val_y = preprocess_df(validation_main_df)

    model = get_model(LEARNING_RATE, train_x.shape[1:])

    train_model(model, train_x, train_y, (val_x, val_y), batch_size=BATCH_SIZE, epochs=EPOCHS, model_name=NAME, tb=True)



if __name__ == '__main__':
    main()
