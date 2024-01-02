import base64
from datetime import datetime
import boto3
from PIL import Image
import io
import numpy as np
import onnxruntime as ort
def load_sequence_model(bucket_name, date=None):
    """if date is None:
        date = datetime.now().strftime('%b-%d')
    s3_client = boto3.client('s3')
    object_key = f'Model-{date}.onnx'
    local_model_path = f'Model-{date}.onnx'

    s3_client.download_file(bucket_name, object_key, local_model_path)
    session = ort.InferenceSession(local_model_path)
    return session"""
    if date is None:
        date = datetime.now()

    s3_client = boto3.client('s3')

    # Retry logic for finding the most recent model
    for i in range(12):  # Retry up to 12 months back
        formatted_date = date.strftime('%b-%Y') #Model-Sequential-Tom-Dec-2023.onnx
        object_key = f'Model-Sequential-Tom-{formatted_date}.onnx'
        local_model_path = f'Model-Sequential-{formatted_date}.onnx'

        try:
            s3_client.download_file(bucket_name, object_key, local_model_path)
            session = ort.InferenceSession(local_model_path)
            return session
        except s3_client.exceptions.NoSuchKey:
            # Decrement the month manually, adjust the year if needed
            month = date.month - 1
            year = date.year
            if month == 0:  # If January, move to December of previous year
                month = 12
                year -= 1
            date = date.replace(month=month, year=year)
            continue

    raise Exception("Model not found for the past 12 months.")

class WeatherPredictionSequenceModel(object):
    def __init__(self):
        self.ort_session = load_sequence_model("austin-weather-prediction-models")
        self.input_names = [input.name for input in self.ort_session.get_inputs()]

    def predict(self, *args):
        data = args[0]
        seq_len = 20
        single_image_size = 512 * 512 * 3
        single_date_size = 4  # Assuming each date is represented by 4 features

        if len(data) == (single_image_size + single_date_size) * seq_len:
            # Reshape the input data to accommodate the sequence of 20 images and dates
            images = []
            dates = []

            for i in range(seq_len):
                start_idx = i * (single_image_size + single_date_size)
                end_idx_image = start_idx + single_image_size
                end_idx_date = end_idx_image + single_date_size

                data_image = data[start_idx:end_idx_image]
                data_date = data[end_idx_image:end_idx_date]

                image_array = np.array(data_image).reshape((512, 512, 3))
                image_array = np.transpose(image_array, (2, 0, 1))  # (C, H, W)
                image_array = image_array.astype(np.float32) / 255.0  # Normalize if needed

                images.append(image_array)
                dates.append(np.array(data_date).astype(np.float32))

            # Stack images and dates to form batch dimensions
            image_batch = np.expand_dims(np.stack(images, axis=0), axis=0)  # (1, N, C, H, W)
            date_batch = np.expand_dims(np.stack(dates, axis=0), axis=0)  # (1, N, D) where D is the date feature size

            onnx_inputs = {self.input_names[0]: image_batch, self.input_names[1]: date_batch}

            # Process the sequence to make a prediction
            onnx_output = self.ort_session.run(None, onnx_inputs)
            # Assuming the output is a sequence of predictions
            predictions = onnx_output[0]

            # Process and return the predictions as needed
            return predictions.tolist()
        else:
            raise ValueError(f"Input data malformed, received length {len(data)}")


