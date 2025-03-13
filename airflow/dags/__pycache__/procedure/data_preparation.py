import pandas as pd
from statsmodels.tsa.stattools import adfuller

def prepare_data(df):
    # Ensure 'Order Date' is a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df = df.set_index('Order Date')

    # Resample data to monthly frequency
    df_resampled = df.resample('ME')['Total Sales'].sum()

    # Check for stationarity and difference if necessary
    result = adfuller(df_resampled)
    df_stationary = df_resampled
    if result[1] > 0.05:
        df_stationary = df_resampled.diff().dropna()
        result = adfuller(df_stationary)
        if result[1] > 0.05:
            df_stationary = df_stationary.diff().dropna()

    # Split data into training and testing sets
    train_size = int(len(df_stationary) * 0.8)
    df_train = df_stationary[:train_size]
    df_test = df_stationary[train_size:]

    return df_train, df_test