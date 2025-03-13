import matplotlib.pyplot as plt

def plot_actual_vs_predicted(df_test, y_test_unscaled, lstm_predictions, lookback):
    plt.figure(figsize=(12, 6))
    plt.plot(df_test.index[lookback:], y_test_unscaled, label='Actual Total Sales', color='blue')
    plt.plot(df_test.index[lookback:], lstm_predictions, label='Predicted Total Sales', color='red')
    plt.xlabel('Order Date')
    plt.ylabel('Total Sales')
    plt.title('Actual vs. Predicted Total Sales (Optimized LSTM)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_residuals(df_test, y_test_unscaled, lstm_predictions, lookback):
    residuals = y_test_unscaled - lstm_predictions
    plt.figure(figsize=(12, 6))
    plt.plot(df_test.index[lookback:], residuals, label='Residuals', color='green')
    plt.xlabel('Order Date')
    plt.ylabel('Residuals')
    plt.title('Residuals of Optimized LSTM Model')
    plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at zero
    plt.legend()
    plt.grid(True)
    plt.show()