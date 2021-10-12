"""Este módulo é responsável por pegar os dados do terminal do metatrader."""
import pytz

import pandas as pd
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

from datetime import datetime

def get_data(currency_pair: str) -> pd.DataFrame:
    """Pega os dados do terminal do metatrader"""
    if not mt5.initialize():
        print('Something went wrong while connecting to the MetaTrader 5 Terminal')
        mt5.shutdown()
    
    ticks = mt5.copy_ticks_from(currency_pair, datetime(2010,1,1), 10000000, mt5.COPY_TICKS_ALL)
        
    mt5.shutdown()

    return pd.DataFrame(ticks)

def get_rates(currency_pair: str) -> pd.DataFrame:
    """Pega os dados em barras do terminal do metatrader."""
    if not mt5.initialize():
        print('Something went wrong while connecting to the MetaTrader 5 Terminal')
        mt5.shutdown()

    timezone = pytz.timezone("Etc/UTC")
    utc_from = datetime(2014, 1, 1, tzinfo=timezone)

    rates = mt5.copy_rates_from(currency_pair, mt5.TIMEFRAME_D1, utc_from, 50000)

    mt5.shutdown()

    return pd.DataFrame(rates)

def main():
    """Pega os dados e salva em um arquivo."""
    data = get_rates('EURUSD')
    data.to_csv('data/eurusd.d1.csv', index=False)

if __name__ == '__main__':
    main()
