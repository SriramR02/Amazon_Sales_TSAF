from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense# type: ignore
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_arima(df_train):
    try:
        arima_model = pm.auto_arima(df_train, seasonal=False, suppress_warnings=True, error_action="ignore")
        arima_aic = arima_model.aic()
        arima_bic = arima_model.bic()
        print(f"ARIMA AIC: {arima_aic}, BIC: {arima_bic}")
        return arima_model
    except Exception as e:
        print(f"An error occurred during ARIMA model training: {e}")
        return None

def train_sarima(df_train):
    try:
        sarima_model = SARIMAX(df_train, order=(5, 0, 1), seasonal_order=(1, 0, 1, 12)).fit(disp=False)
        sarima_aic = sarima_model.aic
        sarima_bic = sarima_model.bic
        print(f"SARIMA AIC: {sarima_aic}, BIC: {sarima_bic}")
        return sarima_model
    except Exception as e:
        print(f"An error occurred during SARIMA model training: {e}")
        return None

def train_optimized_lstm(df_train, df_test, lookback=6, epochs=50, lstm_units=100):
    try:
        # Check if dataframes are empty
        if df_train.empty or df_test.empty:
            print("Error: Training or testing dataset is empty.")
            raise ValueError("Empty dataframe")

        # Scale data
        scaler = MinMaxScaler()
        df_train_scaled = scaler.fit_transform(np.array(df_train).reshape(-1, 1))
        df_test_scaled = scaler.transform(np.array(df_test).reshape(-1, 1))

        # Reshape data
        X_train, y_train = [], []
        for i in range(lookback, len(df_train_scaled)):
            X_train.append(df_train_scaled[i - lookback:i, 0])
            y_train.append(df_train_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        X_test, y_test = [], []
        for i in range(lookback, len(df_test_scaled)):
            X_test.append(df_test_scaled[i - lookback:i, 0])
            y_test.append(df_test_scaled[i, 0])
        X_test, y_test = np.array(X_test), np.array(y_test)

        if len(X_test) == 0:
            print("Error: X_test is empty after reshaping. Check lookback parameter.")
            raise ValueError("Empty X_test")

        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Define and train the model
        optimized_lstm_model = Sequential()
        optimized_lstm_model.add(LSTM(lstm_units, activation='relu', input_shape=(lookback, 1)))
        optimized_lstm_model.add(Dense(1))
        optimized_lstm_model.compile(optimizer='adam', loss='mse')
        optimized_lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0)

        # Make predictions
        lstm_predictions_scaled = optimized_lstm_model.predict(X_test)
        lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
        y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Evaluate the model
        rmse_optimized_lstm = np.sqrt(mean_squared_error(y_test_unscaled, lstm_predictions))
        mae_optimized_lstm = mean_absolute_error(y_test_unscaled, lstm_predictions)
        mape_optimized_lstm = mape(y_test_unscaled, lstm_predictions)
        print(f"Optimized LSTM - RMSE: {rmse_optimized_lstm:.2f}, MAE: {mae_optimized_lstm:.2f}, MAPE: {mape_optimized_lstm:.2f}%")

        return optimized_lstm_model, scaler, lookback

    except Exception as e:
        print(f"An error occurred during optimized LSTM model training or evaluation: {e}")
        return None, None, None
    

# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import pmdarima as pm
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# import numpy as np
    # def mape(y_true, y_pred):
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# def train_arima(df_train):
#     try:
#         arima_model = pm.auto_arima(df_train, seasonal=False, suppress_warnings=True, error_action="ignore")
#         arima_aic = arima_model.aic()
#         arima_bic = arima_model.bic()
#         print(f"ARIMA AIC: {arima_aic}, BIC: {arima_bic}")
#         return arima_model
#     except Exception as e:
#         print(f"An error occurred during ARIMA model training: {e}")
#         return None

# def train_sarima(df_train):
#     try:
#         sarima_model = SARIMAX(df_train, order=(5, 0, 1), seasonal_order=(1, 0, 1, 12)).fit(disp=False)
#         sarima_aic = sarima_model.aic
#         sarima_bic = sarima_model.bic
#         print(f"SARIMA AIC: {sarima_aic}, BIC: {sarima_bic}")
#         return sarima_model
#     except Exception as e:
#         print(f"An error occurred during SARIMA model training: {e}")
#         return None

# def train_optimized_lstm(df_train, df_test, lookback=6, epochs=50, lstm_units=100):
#     try:
#         # Check if dataframes are empty
#         if df_train.empty or df_test.empty:
#             print("Error: Training or testing dataset is empty.")
#             raise ValueError("Empty dataframe")

#         # Scale data
#         scaler = MinMaxScaler()
#         df_train_scaled = scaler.fit_transform(np.array(df_train).reshape(-1, 1))
#         df_test_scaled = scaler.transform(np.array(df_test).reshape(-1, 1))

#         # Reshape data
#         X_train, y_train = [], []
#         for i in range(lookback, len(df_train_scaled)):
#             X_train.append(df_train_scaled[i - lookback:i, 0])
#             y_train.append(df_train_scaled[i, 0])
#         X_train, y_train = np.array(X_train), np.array(y_train)
#         X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

#         X_test, y_test = [], []
#         for i in range(lookback, len(df_test_scaled)):
#             X_test.append(df_test_scaled[i - lookback:i, 0])
#             y_test.append(df_test_scaled[i, 0])
#         X_test, y_test = np.array(X_test), np.array(y_test)

#         if len(X_test) == 0:
#             print("Error: X_test is empty after reshaping. Check lookback parameter.")
#             raise ValueError("Empty X_test")

#         X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#         # Define and train the model
#         optimized_lstm_model = Sequential()
#         optimized_lstm_model.add(LSTM(lstm_units, activation='relu', input_shape=(lookback, 1)))
#         optimized_lstm_model.add(Dense(1))
#         optimized_lstm_model.compile(optimizer='adam', loss='mse')
#         optimized_lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0)

#         # Make predictions
#         lstm_predictions_scaled = optimized_lstm_model.predict(X_test)
#         lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
#         y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

#         # Evaluate the model
#         rmse_optimized_lstm = np.sqrt(mean_squared_error(y_test_unscaled, lstm_predictions))
#         mae_optimized_lstm = mean_absolute_error(y_test_unscaled, lstm_predictions)
#         mape_optimized_lstm = mape(y_test_unscaled, lstm_predictions)
#         print(f"Optimized LSTM - RMSE: {rmse_optimized_lstm:.2f}, MAE: {mae_optimized_lstm:.2f}, MAPE: {mape_optimized_lstm:.2f}%")

#         return optimized_lstm_model, scaler, lookback

#     except Exception as e:
#         print(f"An error occurred during optimized LSTM model training or evaluation: {e}")
#         return None, None, None