# Use an official Python runtime as a parent image
FROM python:3.10.5
# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt 

# Make port 80 available to the world outside this container
EXPOSE 80


ENV NAME=docker

# Run main.py when the container launches
CMD ["python", "main.py"]