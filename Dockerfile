FROM python:3.9-slim
WORKDIR /app

# Install python packages
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Port for GRPC
EXPOSE 5000
# Port for REST
EXPOSE 9000

# Define environment variables WeatherPredictionModel
ENV MODEL_NAME WeatherPredictionSequenceModel
ENV SERVICE_TYPE MODEL


# Changing folder to default user
RUN chown -R 8888 /app

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE
