import base64
from datetime import datetime
import boto3
from PIL import Image
import io
import numpy as np
import onnxruntime as ort

def load_model( bucket_name, date=None):
    if date is None:
        date = datetime.now().strftime('%b-%d')
    s3_client = boto3.client('s3')
    object_key = f'Model-{date}.onnx'
    local_model_path = f'Model-{date}.onnx'

    s3_client.download_file(bucket_name, object_key, local_model_path)
    session = ort.InferenceSession(local_model_path)
    return session

class WeatherPredictionModel(object):
    def __init__(self):
        self.ort_session = load_model("austin-weather-prediction-models", date='Nov-01')
        self.input_names = [input.name for input in self.ort_session.get_inputs()]
    def predict(self, *args):
        data = args[0]
        if len(data) == 786436:
            data_image = data[0:512 * 512 * 3]
            data_date = data[512 * 512 * 3:]
            image_array = np.array(data_image).reshape((512, 512, 3))
            # Ensure the image has the shape (C, H, W) as expected by ONNX
            image_array = np.transpose(image_array, (2, 0, 1))
            # Add a batch dimension (N, C, H, W) and convert to float32
            image_array = image_array.astype(np.float32) / 255.0  # Normalize if needed
            image_array = np.expand_dims(image_array, axis=0)
            # Use the float values (features) as needed
            # Assuming features is a list of 4 float values
            features_array = np.array(data_date).astype(np.float32).reshape(1, -1)
            onnx_inputs = {self.input_names[0]: image_array, self.input_names[1]: features_array}
            # Process the image and features to make a prediction
            onnx_output = self.ort_session.run(None, onnx_inputs)
            pred_high, pred_low = onnx_output[0][0][0], onnx_output[0][0][1]
            # Return the result
            return [float(pred_high), float(pred_low)]
        else:
            raise ValueError(f"Input data malformed, received {args}")