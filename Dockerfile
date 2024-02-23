# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr

# Copy the current directory contents into the container at /usr/src/app
COPY /data/regression/*.csv /usr/data/regression/
COPY /data/regression/reg_w_preds_1.p /usr/data/regression/reg_w_preds_1.p
COPY /src/util/logger_config.py ./src/util/logger_config.py
COPY /src/modules/simQuery/simulationQuery.py ./src/modules/simQuery/simulationQuery.py
COPY /src/modules/simQuery/app.py ./src/modules/simQuery/app.py
COPY /src/modules/simQuery/__init__.py ./src/modules/simQuery/__init__.py
COPY /src/domain/common.py ./src/domain/common.py
COPY /requirements-app.txt /usr/requirements-app.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r ./requirements-app.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=src.modules.simQuery.app
ENV FLASK_RUN_HOST=0.0.0.0

# Run app.py when the container launches
CMD ["flask", "run"]