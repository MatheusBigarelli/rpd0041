"""The porpouse of this notebook is to choose the appropriate input length for the LSTM model based on the accuracy of cross-validation for each input length tested.

The input lengths that will be tested will be half, equal and double the number of ahead-minutes of the horizon (5). So, the lenghts will be:

 - 2 (floor(5/2))
 - 5
 - 10
As sub-tasks, there are:

 1. Create each dataset with the respective time window for the model
 2. Train and evaluate the model given the dataset
 3. Select time window of best results"""


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers.core import SlicingOpLambda

from util import Baseline, WindowGenerator


def compile_and_fit(model, window, patience=2, max_epochs=10):
    

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min'
    )

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanSquaredError()])

    history = model.fit(window.train, epochs=max_epochs, validation_data=window.val)
    return history


def create_window(df, input_width):
    """Create a window generator using a dataset with input and label width
    specified."""

    train_pct = 0.7
    val_pct = 0.2

    total_size = df.shape[0]

    train_slice = slice(0, int(train_pct*total_size))
    val_slice = slice(int(train_pct*total_size), int((train_pct + val_pct)*total_size))
    test_slice = slice(int((train_pct + val_pct)*total_size), None)

    window = WindowGenerator(
        input_width=input_width, label_width=input_width, shift=5,
        label_columns=['direction'],
        train_df=df[train_slice],
        val_df=df[val_slice],
        test_df=df[test_slice],
    )

    return window


def experiment1(df):
    models = [
        Sequential([
            tf.keras.layers.LSTM(16, return_sequences=True),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ]),
        
        Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ]),
        
        Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ]),

        Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ]),
    ]

    x = []
    y = []

    _, ax = plt.subplots(3, 1, figsize=(12, 8))

    for i, input_width in enumerate(range(5, 20, 5)):

        for lstm_model in models:

            window = create_window(df, input_width)

            print('Input shape:', window.example[0].shape)
            print('Output shape:', lstm_model(window.example[0]).shape)

            history = compile_and_fit(lstm_model, window)

            print(f'Performance: {lstm_model.evaluate(window.test)}')

            x.append(input_width)
            y.append(lstm_model.evaluate(window.test)[1])

        ax[i].plot(x, y, label=f'input_width: {input_width}')

    plt.show()


def experiment2(df):
    models = [
        Sequential([
            tf.keras.layers.LSTM(16, return_sequences=True),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ]),
        
        Sequential([
            tf.keras.layers.LSTM(16, return_sequences=True),
            tf.keras.layers.LSTM(16, return_sequences=True),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ]),
        
        Sequential([
            tf.keras.layers.LSTM(16, return_sequences=True),
            tf.keras.layers.LSTM(16, return_sequences=True),
            tf.keras.layers.LSTM(16, return_sequences=True),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ]),

        Sequential([
            tf.keras.layers.LSTM(16, return_sequences=True),
            tf.keras.layers.LSTM(16, return_sequences=True),
            tf.keras.layers.LSTM(16, return_sequences=True),
            tf.keras.layers.LSTM(16, return_sequences=True),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ]),
    ]

    x = []
    y = []

    _, ax = plt.subplots(3, 1, figsize=(12, 8))

    for i, input_width in enumerate(range(5, 20, 5)):

        for lstm_model in models:

            window = create_window(df, input_width)

            print('Input shape:', window.example[0].shape)
            print('Output shape:', lstm_model(window.example[0]).shape)

            history = compile_and_fit(lstm_model, window)

            print(f'Performance: {lstm_model.evaluate(window.test)}')

            x.append(input_width)
            y.append(lstm_model.evaluate(window.test)[1])

        ax[i].plot(x, y, label=f'input_width: {input_width}')

    plt.show()


def experiment3(df):
    models = [
        Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ]),
        
        Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ]),
        
        Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ]),

        Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ]),
    ]

    x = []
    y = []

    _, ax = plt.subplots(3, 1, figsize=(12, 8))

    for i, input_width in enumerate(range(5, 20, 5)):

        for lstm_model in models:

            window = create_window(df, input_width)

            print('Input shape:', window.example[0].shape)
            print('Output shape:', lstm_model(window.example[0]).shape)

            history = compile_and_fit(lstm_model, window)

            print(f'Performance: {lstm_model.evaluate(window.test)}')

            x.append(input_width)
            y.append(lstm_model.evaluate(window.test)[1])

        ax[i].plot(x, y, label=f'input_width: {input_width}')

    plt.show()


def experiment4(df):
    models = [
        Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ]),
        
        Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ]),
        
        Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ]),

        Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ]),
    ]

    x = []
    y = []

    _, ax = plt.subplots(3, 1, figsize=(12, 8))

    for i, input_width in enumerate(range(5, 20, 5)):

        for lstm_model in models:

            window = create_window(df, input_width)

            print('Input shape:', window.example[0].shape)
            print('Output shape:', lstm_model(window.example[0]).shape)

            history = compile_and_fit(lstm_model, window)

            print(f'Performance: {lstm_model.evaluate(window.test)}')

            x.append(input_width)
            y.append(lstm_model.evaluate(window.test)[1])

        ax[i].plot(x, y, label=f'input_width: {input_width}')

    plt.show()


def experiment5(df):
    models = [
        Sequential([
            tf.keras.layers.Dense(16, activation='sigmoid'),
            tf.keras.layers.Dense(1)
        ]),
        
        Sequential([
            tf.keras.layers.Dense(32, activation='sigmoid'),
            tf.keras.layers.Dense(1)
        ]),
        
        Sequential([
            tf.keras.layers.Dense(64, activation='sigmoid'),
            tf.keras.layers.Dense(1)
        ]),

        Sequential([
            tf.keras.layers.Dense(128, activation='sigmoid'),
            tf.keras.layers.Dense(1)
        ]),
    ]

    x = []
    y = []

    _, ax = plt.subplots(3, 1, figsize=(12, 8))

    for i, input_width in enumerate(range(5, 20, 5)):

        for lstm_model in models:

            window = create_window(df, input_width)

            print('Input shape:', window.example[0].shape)
            print('Output shape:', lstm_model(window.example[0]).shape)

            history = compile_and_fit(lstm_model, window)

            print(f'Performance: {lstm_model.evaluate(window.test)}')

            x.append(input_width)
            y.append(lstm_model.evaluate(window.test)[1])

        ax[i].plot(x, y, label=f'input_width: {input_width}')

    plt.show()


def experiment6(df):
    models = [
        Sequential([
            tf.keras.layers.Dense(16, activation='sigmoid'),
            tf.keras.layers.Dense(1)
        ]),
        
        Sequential([
            tf.keras.layers.Dense(16, activation='sigmoid'),
            tf.keras.layers.Dense(1)
        ]),
        
        Sequential([
            tf.keras.layers.Dense(16, activation='sigmoid'),
            tf.keras.layers.Dense(1)
        ]),

        Sequential([
            tf.keras.layers.Dense(16, activation='sigmoid'),
            tf.keras.layers.Dense(1)
        ]),
    ]

    _, ax = plt.subplots(3, len(models), figsize=(12, 8))

    for i, input_width in enumerate(range(5, 20, 5)):

        for j, lstm_model in enumerate(models):

            x = []
            y = []
            for epochs in range(10, 30, 10):

                window = create_window(df, input_width)

                print('Input shape:', window.example[0].shape)
                print('Output shape:', lstm_model(window.example[0]).shape)

                history = compile_and_fit(lstm_model, window, epochs)

                print(f'Performance: {lstm_model.evaluate(window.test)}')

                x.append(input_width)
                y.append(lstm_model.evaluate(window.test)[1])

            ax[i, j].plot(x, y, label=f'{input_width}-{epochs}')

    plt.legend()
    plt.show()


def experiment7(df):
    models = [
        Sequential([
            tf.keras.layers.Dense(1)
        ]),
        
        Sequential([
            tf.keras.layers.Dense(1, activation='relu')
        ]),
        
        Sequential([
            tf.keras.layers.Dense(1, activation='sigmoid')
        ]),

        Sequential([
            tf.keras.layers.Dense(1, activation='tanh')
        ]),
    ]

    x = []
    y = []

    _, ax = plt.subplots(3, 1, figsize=(12, 8))

    for i, input_width in enumerate(range(5, 20, 5)):

        for lstm_model in models:

            window = create_window(df, input_width)

            print('Input shape:', window.example[0].shape)
            print('Output shape:', lstm_model(window.example[0]).shape)

            history = compile_and_fit(lstm_model, window)

            print(f'Performance: {lstm_model.evaluate(window.test)}')

            x.append(input_width)
            y.append(lstm_model.evaluate(window.test)[1])

        ax[i].plot(x, y, label=f'input_width: {input_width}')

    plt.show()


def experiment8(df):
    lstm_model = Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    x = []
    y = []

    _, ax = plt.subplots(3, 1, figsize=(12, 8))

    for i, input_width in enumerate(range(5, 20, 5)):
        window = create_window(df, input_width)

        history = compile_and_fit(lstm_model, window)

        print(f'Performance: {lstm_model.evaluate(window.test)}')

        x.append(input_width)
        y.append(lstm_model.evaluate(window.test)[1])

    plt.plot(x, y, label=f'input_width: {input_width}')
    plt.show()


def main():
    df_ilo = pd.read_csv('data/ilo.csv', index_col='time')

    # window1 = create_window(df_ilo, 3)
    # window2 = create_window(df_ilo, 5)
    # window3 = create_window(df_ilo, 10)

    # windows = [window1, window2, window3]


    # experiment1(df_ilo)
    # experiment2(df_ilo)
    # experiment3(df_ilo)
    # experiment4(df_ilo)
    # experiment5(df_ilo)
    # experiment6(df_ilo)
    # experiment7(df_ilo)
    experiment8(df_ilo)
    

if __name__ == '__main__':
    main()
