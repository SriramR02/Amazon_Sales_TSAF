from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_arima(arima_model, df_test):
    arima_predictions = arima_model.predict(n_periods=len(df_test))
    rmse_arima = np.sqrt(mean_squared_error(df_test, arima_predictions))
    mae_arima = mean_absolute_error(df_test, arima_predictions)
    mape_arima = mape(df_test, arima_predictions)
    print(f"ARIMA - RMSE: {rmse_arima:.2f}, MAE: {mae_arima:.2f}, MAPE: {mape_arima:.2f}%")
    print(f"ARIMA - AIC: {arima_model.aic()}, BIC: {arima_model.bic()}")

def evaluate_sarima(sarima_model, df_train, df_test):
    try:
        sarima_predictions = sarima_model.predict(start=len(df_train), end=len(df_train) + len(df_test) - 1, dynamic=False)
        rmse_sarima = np.sqrt(mean_squared_error(df_test, sarima_predictions))
        mae_sarima = mean_absolute_error(df_test, sarima_predictions)
        mape_sarima = mape(df_test, sarima_predictions)
        print(f"SARIMA - RMSE: {rmse_sarima:.2f}, MAE: {mae_sarima:.2f}, MAPE: {mape_sarima:.2f}%")
        print(f"SARIMA - AIC: {sarima_model.aic}, BIC: {sarima_model.bic}")
    except Exception as e:
        print(f"An error occurred during SARIMA prediction: {e}")

def evaluate_lstm(lstm_model, scaler, df_test, lookback=12):
    try:
        df_test_scaled = scaler.transform(np.array(df_test).reshape(-1, 1))
        X_test, y_test = [], []
        for i in range(lookback, len(df_test_scaled)):
            X_test.append(df_test_scaled[i-lookback:i, 0])
            y_test.append(df_test_scaled[i, 0])
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        lstm_predictions_scaled = lstm_model.predict(X_test)
        lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
        y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse_lstm = np.sqrt(mean_squared_error(y_test_unscaled, lstm_predictions))
        mae_lstm = mean_absolute_error(y_test_unscaled, lstm_predictions)
        mape_lstm = mape(y_test_unscaled, lstm_predictions)
        print(f"LSTM - RMSE: {rmse_lstm:.2f}, MAE: {mae_lstm:.2f}, MAPE: {mape_lstm:.2f}%")
    except Exception as e:
        print(f"An error occurred during LSTM prediction: {e}")