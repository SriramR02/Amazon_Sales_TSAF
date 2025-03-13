import pandas as pd
import numpy as np
from data_processing.load_data import load_data
from data_processing.clean_data import clean_data
from data_preparation import prepare_data
from modeling import train_optimized_lstm,train_arima,train_sarima
from evaluation import evaluate_arima, evaluate_sarima
from visualization import plot_actual_vs_predicted, plot_residuals

def main():
    file_path = './datasets/amazon_sales_dataset_2019_2024_corrected.xlsx'
    df = load_data(file_path)
    if df is not None:
        df = clean_data(df)
        if df is not None:
            print("Data loaded and cleaned successfully.")
            df_train, df_test = prepare_data(df)

            # Train and evaluate ARIMA model
            arima_model = train_arima(df_train)
            if arima_model is not None:
                evaluate_arima(arima_model, df_test)

            # Train and evaluate SARIMA model
            sarima_model = train_sarima(df_train)
            if sarima_model is not None:
                evaluate_sarima(sarima_model, df_train, df_test)

            # Train and evaluate LSTM model
            optimized_lstm_model, scaler, lookback = train_optimized_lstm(df_train, df_test)
            if optimized_lstm_model is not None:
                # Make predictions and visualize results
                df_test_scaled = scaler.transform(np.array(df_test).reshape(-1, 1))
                X_test, y_test = [], []
                for i in range(lookback, len(df_test_scaled)):
                    X_test.append(df_test_scaled[i - lookback:i, 0])
                    y_test.append(df_test_scaled[i, 0])
                X_test, y_test = np.array(X_test), np.array(y_test)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                
                lstm_predictions_scaled = optimized_lstm_model.predict(X_test)
                lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
                y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

                plot_actual_vs_predicted(df_test, y_test_unscaled, lstm_predictions, lookback)
                plot_residuals(df_test, y_test_unscaled, lstm_predictions, lookback)
            else:
                print("Model training failed.")
        else:
            print("Data cleaning failed.")
    else:
        print("Data loading failed.")

if __name__ == "__main__":
    main()


# from procedure.data_processing.load_data import load_data
# from procedure.data_processing.clean_data import clean_data
# from procedure.data_preparation import prepare_data
# from procedure.modeling import train_optimized_lstm, train_arima, train_sarima
# from procedure.evaluation import evaluate_arima, evaluate_sarima
# from procedure.visualization import plot_actual_vs_predicted, plot_residuals

# def main():
#     file_path = './datasets/amazon_sales_dataset_2019_2024_corrected.xlsx'
#     df = load_data(file_path)
#     if df is not None:
#         df = clean_data(df)
#         if df is not None:
#             print("Data loaded and cleaned successfully.")
#             df_train, df_test = prepare_data(df)
#             optimized_lstm_model, scaler, lookback = train_optimized_lstm(df_train, df_test)
#             if optimized_lstm_model is not None:
#                 # Make predictions and visualize results
#                 df_test_scaled = scaler.transform(np.array(df_test).reshape(-1, 1))
#                 X_test, y_test = [], []
#                 for i in range(lookback, len(df_test_scaled)):
#                     X_test.append(df_test_scaled[i - lookback:i, 0])
#                     y_test.append(df_test_scaled[i, 0])
#                 X_test, y_test = np.array(X_test), np.array(y_test)
#                 X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                
#                 lstm_predictions_scaled = optimized_lstm_model.predict(X_test)
#                 lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
#                 y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

#                 plot_actual_vs_predicted(df_test, y_test_unscaled, lstm_predictions, lookback)
#                 plot_residuals(df_test, y_test_unscaled, lstm_predictions, lookback)
#             else:
#                 print("Model training failed.")
#         else:
#             print("Data cleaning failed.")
#     else:
#         print("Data loading failed.")

# if __name__ == "__main__":
#     main()



