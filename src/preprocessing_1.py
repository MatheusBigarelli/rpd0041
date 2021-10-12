""""""
# Tecnhical analysis libraries
from ta.trend import EMAIndicator
from ta.trend import MACD
from ta.momentum import RSIIndicator

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler


def label(df: pd.DataFrame) -> pd.Series:
    """Label the data with the following criterion.
    If the close value 5 minutes ahead is greater the close value of the
    current minute, the label is 1. Otherwise, the label is 0

    :param:df:pd.DataFrame: The data that must be labeled.
    """
    # We take the difference series for the 5 minute ahead and current
    # minute
    difference = df['close'].shift(-5) - df['close']

    # And apply the decision
    return difference.apply(lambda value: 1 if value > 0 else 0).rename('label')


def compute_pct_change(series: pd.Series) -> pd.Series:
    """Computes the percentage change (or return) of the passed series.

    :param:df:pd.DataFrame: The series of which is desired the series of
    returns.
    :return:pd.Series: The series of percentage changes (returns).
    """
    return (series - series.shift(1)) / series.shift(1)


def direction(series: pd.Series) -> pd.Series:
    """Computes the direction of the candles.
    
    :param:series:pd.Series: The data whose direction is wanted.
    :return:pd.Series: The associated directions.
    """
    diffs = series - series.shift(1)

    return diffs.apply(lambda x: 1 if x > 0 else 0)


def main2():
    df = pd.read_csv(
        'data/eurusd.h1.csv',
        index_col='time',
        parse_dates=['time'],
        date_parser=lambda d: pd.to_datetime(d, unit='s')
    )
    df = df[['close']]
    train_pct = 0.8
    split_index = int(df.shape[0] * train_pct)
    df_train = df[:split_index].copy()
    df_test = df[split_index:].copy()

    df_train['direction'] = direction(df_train['close'])
    df_test['direction'] = direction(df_test['close'])

    df_train['label'] = label(df_train)
    df_test['label'] = label(df_test)

    # ---------------------------------------------------------------
    # Training set
    # EMAs
    # 12
    df_train['ema12'] = EMAIndicator(df_train['close'], window=12).ema_indicator()
    df_train['ema26'] = EMAIndicator(df_train['close'], window=26).ema_indicator()
    # df_train['ema50'] = EMAIndicator(df_train['close'], window=50).ema_indicator()
    # df_train['ema200'] = EMAIndicator(df_train['close'], window=200).ema_indicator()

    # MACD
    macd = MACD(df_train['close'])
    df_train['macd_value'] = macd.macd()
    df_train['macd_histogram'] = macd.macd_diff()
    df_train['macd_signal'] = macd.macd_signal()

    # RSI
    df_train['rsi'] = RSIIndicator(df_train['close']).rsi()

    # ---------------------------------------------------------------
    # Testing set
    # EMAs
    # 12
    df_test['ema12'] = EMAIndicator(df_test['close'], window=12).ema_indicator()
    df_test['ema26'] = EMAIndicator(df_test['close'], window=26).ema_indicator()

    # MACD
    macd = MACD(df_test['close'])
    df_test['macd_value'] = macd.macd()
    df_test['macd_histogram'] = macd.macd_diff()
    df_test['macd_signal'] = macd.macd_signal()

    # RSI
    df_test['rsi'] = RSIIndicator(df_test['close']).rsi()

    # ---------------------------------------------------------------

    df_train['ema-d'] = (df_train['ema12'] - df_train['ema26']).apply(lambda x: 1 if x > 0 else 0)
    df_test['ema-d'] = (df_test['ema12'] - df_test['ema26']).apply(lambda x: 1 if x > 0 else 0)

    df_train['rsi-d'] = df_train['rsi'].apply(lambda x: 1
                              if 50 < x < 70 or x < 30
                              else 0)
    df_test['rsi-d'] = df_test['rsi'].apply(lambda x: 1
                              if 50 < x < 70 or x < 30
                              else 0)

    df_train['macd-d'] = df_train.apply(lambda x: 1 if x['macd_signal'] < x['macd_value'] else 0, axis=1)
    df_test['macd-d'] = df_test.apply(lambda x: 1 if x['macd_signal'] < x['macd_value'] else 0, axis=1)

    # df_train_final = df_train[['direction', 'ema-d', 'rsi-d', 'macd-d', 'label']]
    # df_test_final = df_test[['direction', 'ema-d', 'rsi-d', 'macd-d', 'label']]

    cols = [
        'close',
        'ema12',
        'ema26',
        'macd_value',
        'macd_histogram',
        'macd_value',
        'rsi'
    ]
    for col in cols:
        df_train[f'{col}_pctchange'] = compute_pct_change(df_train[col])
        df_test[f'{col}_pctchange'] = compute_pct_change(df_test[col])

    scaler = StandardScaler()

    X_train = df_train.drop(['label'], axis=1)
    y_train = df_train['label']

    X_train_scaled = scaler.fit_transform(X_train)

    X_test = df_test.drop(['label'], axis=1)
    y_test = df_test['label']

    X_test_scaled = scaler.transform(X_test)

    df_train_final = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    df_train_final['label'] = y_train

    df_test_final = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    df_test_final['label'] = y_test

    df_train_final = df_train_final.dropna()
    df_test_final = df_test_final.dropna()
    df_train_final.to_csv('data/train.csv')
    df_train_final.to_csv('data/test.csv')
    split_index = int(df_train_final.shape[0] * 0.5)
    input_length_optimization_dataset = df_train_final[:split_index].copy()
    model_architecture_optimization_dataset = df_train_final[split_index:].copy()
    input_length_optimization_dataset.to_csv('data/ilo.csv')
    model_architecture_optimization_dataset.to_csv('data/mao.csv')


def main():
    df = pd.read_csv(
        'data/eurusd.m1.1.csv',
        index_col='time',
        parse_dates=['time'],
        date_parser=lambda d: pd.to_datetime(d, unit='s')
    )
    df = df[['close']]
    train_pct = 0.8
    split_index = int(df.shape[0] * train_pct)
    df_train = df[:split_index].copy()
    df_test = df[split_index:].copy()
    df_train.to_csv('data/raw_train.csv')
    df_test.to_csv('data/raw_test.csv')

    df_train['direction'] = direction(df_train['close'])
    df_test['direction'] = direction(df_test['close'])

    df_train['label'] = label(df_train)
    df_test['label'] = label(df_test)

    # ---------------------------------------------------------------
    # Training set
    # EMAs
    # 12
    df_train['ema12'] = EMAIndicator(df_train['close'], window=12).ema_indicator()
    df_train['ema26'] = EMAIndicator(df_train['close'], window=26).ema_indicator()

    # MACD
    macd = MACD(df_train['close'])
    df_train['macd_value'] = macd.macd()
    df_train['macd_histogram'] = macd.macd_diff()
    df_train['macd_signal'] = macd.macd_signal()

    # RSI
    df_train['rsi'] = RSIIndicator(df_train['close']).rsi()

    # ---------------------------------------------------------------
    # Testing set
    # EMAs
    # 12
    df_test['ema12'] = EMAIndicator(df_test['close'], window=12).ema_indicator()
    df_test['ema26'] = EMAIndicator(df_test['close'], window=26).ema_indicator()

    # MACD
    macd = MACD(df_test['close'])
    df_test['macd_value'] = macd.macd()
    df_test['macd_histogram'] = macd.macd_diff()
    df_test['macd_signal'] = macd.macd_signal()

    # RSI
    df_test['rsi'] = RSIIndicator(df_test['close']).rsi()

    cols = [
        'close',
        'ema12',
        'ema26',
        'macd_value',
        'macd_histogram',
        'macd_value',
        'rsi'
    ]
    for col in cols:
        df_train[f'{col}_pctchange'] = compute_pct_change(df_train[col])
        df_test[f'{col}_pctchange'] = compute_pct_change(df_test[col])

    scaler = MinMaxScaler()

    X_train = df_train.drop(['label'], axis=1)
    y_train = df_train['label']

    X_train_scaled = scaler.fit_transform(X_train)

    X_test = df_test.drop(['label'], axis=1)
    y_test = df_test['label']

    X_test_scaled = scaler.transform(X_test)

    df_train_final = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    df_train_final['label'] = y_train

    df_test_final = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    df_test_final['label'] = y_test

    df_train_final = df_train_final.dropna()
    df_test_final = df_test_final.dropna()
    df_train_final.to_csv('data/train.csv')
    df_train_final.to_csv('data/test.csv')
    split_index = int(df_train_final.shape[0] * 0.5)
    input_length_optimization_dataset = df_train_final[:split_index].copy()
    model_architecture_optimization_dataset = df_train_final[split_index:].copy()
    input_length_optimization_dataset.to_csv('data/ilo.csv')
    model_architecture_optimization_dataset.to_csv('data/mao.csv')

if __name__ == '__main__':
    main2()
