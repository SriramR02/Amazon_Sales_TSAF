# # Use the official Airflow image as a base
# FROM apache/airflow:2.7.3-python3.10

# # Set the working directory
# WORKDIR /opt/airflow

# # Copy your DAGs and plugins into the image
# COPY dags/ /opt/airflow/dags/
# COPY plugins/ /opt/airflow/plugins/
# COPY datasets/ /opt/airflow/datasets/
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Set Airflow configuration (optional)
# # Example:
# # ENV AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=False

# # Expose Airflow ports (optional, if needed)
# EXPOSE 8081 5555

# # Set the Airflow user and group (optional, if needed)
# USER airflow

# FROM apache/airflow:2.10.5

# # Install any necessary system dependencies
# USER root
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Switch back to airflow user
# USER airflow

# # Copy requirements.txt to the Docker image
# COPY requirements.txt /requirements.txt

# # Install Python dependencies
# RUN pip install --no-cache-dir -r /requirements.txt

# # Copy your DAGs and plugins to the Docker image
# COPY dags/ /opt/airflow/dags/
# COPY plugins/ /opt/airflow/plugins/
# COPY datasets/ /opt/airflow/datasets/


FROM apache/airflow:2.10.5-python3.10

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt