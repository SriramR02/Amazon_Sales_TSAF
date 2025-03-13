# import sys
# import os
# from datetime import datetime, timedelta
# from airflow import DAG
# from airflow.operators.python_operator import PythonOperator # type: ignore
# import numpy as np  # Added to resolve errors with numpy.
# import getpass

# # Determine if the procedure folder is in the plugins directory.
# plugins_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plugins')
# procedure_path_plugins = os.path.join(plugins_dir, 'procedure')
# procedure_path_relative = os.path.abspath(os.path.join(os.path.dirname(__file__), 'procedure'))

# if os.path.exists(procedure_path_plugins):
#     sys.path.insert(0, procedure_path_plugins)
# else:
#     sys.path.insert(0, procedure_path_relative)

# # Import your custom modules
# from procedure.data_processing.load_data import load_data
# from procedure.data_processing.clean_data import clean_data
# from procedure.data_preparation import prepare_data
# from procedure.modeling import train_optimized_lstm, train_arima, train_sarima
# from procedure.evaluation import evaluate_arima, evaluate_sarima
# from procedure.visualization import plot_actual_vs_predicted, plot_residuals

# # Get the current user
# user = getpass.getuser()

# # Define your default arguments
# default_args = {
#     'owner': user,
#     'depends_on_past': False,
#     'email_on_failure': False,
#     'email_on_retry': False,
#     'retries': 1,
#     'retry_delay': timedelta(minutes=5),
# }

# dag = DAG(
#     'my_project_dag',
#     default_args=default_args,
#     description='A simple data processing and modeling DAG',
#     schedule_interval='@daily',
#     start_date=datetime.today(),
#     catchup=False,
# )

# def load_data_task():
#     file_path = './datasets/amazon_sales_dataset_2019_2024_corrected.xlsx'
#     print(f"Running as user: {user}")
#     return load_data(file_path)

# def clean_data_task(**context):
#     df = context['task_instance'].xcom_pull(task_ids='load_data')
#     print(f"Running as user: {user}")
#     return clean_data(df)

# def prepare_data_task(**context):
#     df = context['task_instance'].xcom_pull(task_ids='clean_data')
#     print(f"Running as user: {user}")
#     return prepare_data(df)

# def train_models_task(**context):
#     df_train, df_test = context['task_instance'].xcom_pull(task_ids='prepare_data')
#     print(f"Running as user: {user}")
#     arima_model = train_arima(df_train)
#     sarima_model = train_sarima(df_train)
#     optimized_lstm_model, scaler, lookback = train_optimized_lstm(df_train, df_test)
#     return arima_model, sarima_model, optimized_lstm_model, scaler, lookback, df_train, df_test

# def evaluate_models_task(**context):
#     arima_model, sarima_model, optimized_lstm_model, scaler, lookback, df_train, df_test = context['task_instance'].xcom_pull(task_ids='train_models')
#     print(f"Running as user: {user}")
#     evaluate_arima(arima_model, df_test)
#     evaluate_sarima(sarima_model, df_train, df_test)
#     df_test_scaled = scaler.transform(np.array(df_test).reshape(-1, 1))
#     X_test, y_test = [], []
#     for i in range(lookback, len(df_test_scaled)):
#         X_test.append(df_test_scaled[i - lookback:i, 0])
#         y_test.append(df_test_scaled[i, 0])
#     X_test, y_test = np.array(X_test), np.array(y_test)
#     X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
#     lstm_predictions_scaled = optimized_lstm_model.predict(X_test)
#     lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
#     y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
#     plot_actual_vs_predicted(df_test, y_test_unscaled, lstm_predictions, lookback)
#     plot_residuals(df_test, y_test_unscaled, lstm_predictions, lookback)

# load_data_op = PythonOperator(
#     task_id='load_data',
#     python_callable=load_data_task,
#     dag=dag,
# )

# clean_data_op = PythonOperator(
#     task_id='clean_data',
#     python_callable=clean_data_task,
#     provide_context=True,
#     dag=dag,
# )

# prepare_data_op = PythonOperator(
#     task_id='prepare_data',
#     python_callable=prepare_data_task,
#     provide_context=True,
#     dag=dag,
# )

# train_models_op = PythonOperator(
#     task_id='train_models',
#     python_callable=train_models_task,
#     provide_context=True,
#     dag=dag,
# )

# evaluate_models_op = PythonOperator(
#     task_id='evaluate_models',
#     python_callable=evaluate_models_task,
#     provide_context=True,
#     dag=dag,
# )

# load_data_op >> clean_data_op >> prepare_data_op >> train_models_op >> evaluate_models_op

import sys
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator  # type: ignore
import numpy as np  # Added to resolve errors with numpy.
import getpass
import math

# Determine if the procedure folder is in the plugins directory.
plugins_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plugins')
procedure_path_plugins = os.path.join(plugins_dir, 'procedure')
procedure_path_relative = os.path.abspath(os.path.join(os.path.dirname(__file__), 'procedure'))

if os.path.exists(procedure_path_plugins):
    sys.path.insert(0, procedure_path_plugins)
else:
    sys.path.insert(0, procedure_path_relative)

# Import your custom modules
from sklearn.metrics import mean_squared_error
from procedure.load_data import load_data
from procedure.clean_data import clean_data
from procedure.data_preparation import prepare_data
from procedure.modeling import train_optimized_lstm, train_arima, train_sarima
from procedure.evaluation import evaluate_arima, evaluate_sarima
from procedure.visualization import plot_actual_vs_predicted, plot_residuals

# Get the current user
user = getpass.getuser()

# Define your default arguments
default_args = {
    'owner': user,
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'my_project_dag',
    default_args=default_args,
    description='data processing and modeling DAG',
    schedule_interval='@daily',
    start_date=datetime.today(),
    catchup=False,
)

def load_data_task():
    file_path = './datasets/amazon_sales_dataset_2019_2024_corrected.xlsx'
    print(f"Running as user: {user}")
    return load_data(file_path)

def clean_data_task(**context):
    df = context['task_instance'].xcom_pull(task_ids='load_data')
    print(f"Running as user: {user}")
    return clean_data(df)

def prepare_data_task(**context):
    df = context['task_instance'].xcom_pull(task_ids='clean_data')
    print(f"Running as user: {user}")
    return prepare_data(df)

def train_models_task(**context):
    df_train, df_test = context['task_instance'].xcom_pull(task_ids='prepare_data')
    print(f"Running as user: {user}")
    arima_model = train_arima(df_train)
    sarima_model = train_sarima(df_train)
    optimized_lstm_model, scaler, lookback = train_optimized_lstm(df_train, df_test)
    return arima_model, sarima_model, optimized_lstm_model, scaler, lookback, df_train, df_test

def evaluate_arima_task(**context):
    arima_model, _, _, _, _, df_train, df_test = context['task_instance'].xcom_pull(task_ids='train_models')
    print(f"Running as user: {user}")
    arima_predictions = arima_model.predict(start=len(df_train), end=len(df_train) + len(df_test) - 1)
    arima_rmse = math.sqrt(mean_squared_error(df_test, arima_predictions))
    return arima_rmse, arima_predictions

def evaluate_sarima_task(**context):
    _, sarima_model, _, _, _, df_train, df_test = context['task_instance'].xcom_pull(task_ids='train_models')
    print(f"Running as user: {user}")
    sarima_predictions = sarima_model.predict(start=len(df_train), end=len(df_train) + len(df_test) - 1)
    sarima_rmse = math.sqrt(mean_squared_error(df_test, sarima_predictions))
    return sarima_rmse, sarima_predictions

def evaluate_lstm_task(**context):
    _, _, optimized_lstm_model, scaler, lookback, _, df_test = context['task_instance'].xcom_pull(task_ids='train_models')
    print(f"Running as user: {user}")
    df_test_scaled = scaler.transform(np.array(df_test).reshape(-1, 1))
    X_test, y_test = [], []
    for i in range(lookback, len(df_test_scaled)):
        X_test.append(df_test_scaled[i - lookback:i, 0])
        y_test.append(df_test_scaled[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    lstm_predictions_scaled = optimized_lstm_model.predict(X_test)
    lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
    lstm_rmse = math.sqrt(mean_squared_error(y_test.reshape(-1, 1), lstm_predictions))
    return lstm_rmse, lstm_predictions

def visualize_best_model_task(**context):
    arima_rmse, arima_predictions = context['task_instance'].xcom_pull(task_ids='evaluate_arima')
    sarima_rmse, sarima_predictions = context['task_instance'].xcom_pull(task_ids='evaluate_sarima')
    lstm_rmse, lstm_predictions = context['task_instance'].xcom_pull(task_ids='evaluate_lstm')
    _, _, _, _, _, _, df_test = context['task_instance'].xcom_pull(task_ids='train_models')
    
    best_model_name = min(
        [('ARIMA', arima_rmse), ('SARIMA', sarima_rmse), ('LSTM', lstm_rmse)],
        key=lambda x: x[1]
    )[0]
    
    print(f"Best model based on RMSE score: {best_model_name}")
    
    if best_model_name == 'ARIMA':
        plot_actual_vs_predicted(df_test, df_test.values, arima_predictions.values)
        plot_residuals(df_test.values - arima_predictions.values)
        
    elif best_model_name == 'SARIMA':
        plot_actual_vs_predicted(df_test, df_test.values, sarima_predictions.values)
        plot_residuals(df_test.values - sarima_predictions.values)
        
    elif best_model_name == 'LSTM':
        y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        plot_actual_vs_predicted(df_test, y_test_unscaled, lstm_predictions)
        plot_residuals(y_test_unscaled - lstm_predictions)

load_data_op = PythonOperator(
    task_id='load_data',
    python_callable=load_data_task,
    dag=dag,
)

clean_data_op = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data_task,
    provide_context=True,
    dag=dag,
)

prepare_data_op = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data_task,
    provide_context=True,
    dag=dag,
)

train_models_op = PythonOperator(
    task_id='train_models',
    python_callable=train_models_task,
    provide_context=True,
    dag=dag,
)

evaluate_arima_op = PythonOperator(
    task_id='evaluate_arima',
    python_callable=evaluate_arima_task,
    provide_context=True,
    dag=dag,
)

evaluate_sarima_op = PythonOperator(
    task_id='evaluate_sarima',
    python_callable=evaluate_sarima_task,
    provide_context=True,
    dag=dag,
)

evaluate_lstm_op = PythonOperator(
    task_id='evaluate_lstm',
    python_callable=evaluate_lstm_task,
    provide_context=True,
    dag=dag,
)

visualize_best_model_op = PythonOperator(
    task_id='visualize_best_model',
    python_callable=visualize_best_model_task,
    provide_context=True,
    dag=dag,
)

load_data_op >> clean_data_op >> prepare_data_op >> train_models_op >> [evaluate_arima_op, evaluate_sarima_op, evaluate_lstm_op] >> visualize_best_model_op
