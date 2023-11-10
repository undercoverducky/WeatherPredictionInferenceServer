import math

import numpy as np
import json
#from seldon_core.seldon_client import SeldonClient
from owslib.wms import WebMapService
from datetime import datetime, timedelta
import time
import math
import base64
import requests
import io
from PIL import Image
import onnxruntime as ort

# Read NOAA API Token from a file
with open('NOAA_api_key.txt', 'r') as file:
    api_token = file.read().strip()

# NOAA CDO URL
base_url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data'

# Station for Austin, Texas Bergstrom Airport
station_id = 'GHCND:USW00013904'
headers = {
    'token': api_token
}
def fetch_data(date, max_retries=3, backoff_factor=2):
    retries = 0
    while retries < max_retries:
        try:
            params = {
                'datasetid': 'GHCND',  # Global Historical Climatology Network Daily
                'stationid': station_id,
                'startdate': date.strftime('%Y-%m-%d'),
                'enddate': date.strftime('%Y-%m-%d'),
                'units': 'metric',
                'limit': 1000
            }

            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()  # Raise an exception for HTTP error codes
            return response.json()
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            retries += 1
            time.sleep(backoff_factor ** retries)  # Exponential backoff

    raise Exception(f"Failed to fetch data after {max_retries} retries.")

austin_bbox = (-98.2, 29.85, -97.4, 30.7)
gibs_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
wms = WebMapService(gibs_url)
layer = "MODIS_Terra_CorrectedReflectance_TrueColor"
img_format = "image/jpeg"  # Format of the image to retrieve
img_size = (512, 512)  # Size of the image (width, height)
date = datetime.now() - timedelta(days=4)

response = wms.getmap(layers=[layer],
                          styles=[''],
                          srs='EPSG:4326',
                          bbox=austin_bbox,
                          size=img_size,
                          format=img_format,
                          time=date.strftime("%Y-%m-%d"))

encoded_image = base64.b64encode(response.read()).decode('utf-8')
image = Image.open(io.BytesIO(base64.b64decode(encoded_image)))

data = fetch_data(date)
image_array = np.array(image)
print(image_array.shape)
image_list = image_array.flatten().tolist()

# Find the daily high and low temperatures
daily_high_temp = None
daily_low_temp = None
for item in data.get('results', []):
    if item['datatype'] == 'TMAX':
        daily_high_temp = item['value']
    if item['datatype'] == 'TMIN':
        daily_low_temp = item['value']
    if daily_high_temp is not None and daily_low_temp is not None:
        break
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

print(f"ground truth high: {celsius_to_fahrenheit(daily_high_temp)} ground truth low: {celsius_to_fahrenheit(daily_low_temp)}")


day_sin = math.sin(2 * math.pi * date.day / 31)
day_cos = math.cos(2 * math.pi * date.day / 31)
month_sin = math.sin(2 * math.pi * date.month / 12)
month_cos = math.cos(2 * math.pi * date.month / 12)
# We now test the REST endpoint expecting the same result
#print(image_array)
recovered_image = np.array(image_list).reshape((512, 512, 3))
assert np.array_equal(recovered_image, image_array)
image_list.extend([day_sin, day_cos, month_sin, month_cos])

print([day_sin, day_cos, month_sin, month_cos])
batch = {"data": {"ndarray": image_list}}

endpoint = "http://0.0.0.0:9000/api/v1.0/predictions"
headers = {'Content-Type': 'application/json'}

# Make the prediction by sending a POST request
response = requests.post(endpoint, headers=headers, data=json.dumps(batch))
content = json.loads(response.content.decode('utf-8'))
celsius_to_fahrenheit(daily_high_temp)
inference_high = celsius_to_fahrenheit(content["data"]["ndarray"][0])
inference_low = celsius_to_fahrenheit(content['data']["ndarray"][1])

print(f"endpoint high {inference_high} endpoint low {inference_low}")
def load_model(date=None):
    local_model_path = f'Model-{date}.onnx'
    session = ort.InferenceSession(local_model_path)
    return session
ort_session = load_model(date='Nov-01')
image_array = np.transpose(image_array, (2, 0, 1))
# Add a batch dimension (N, C, H, W) and convert to float32
image_array = image_array.astype(np.float32) / 255.0  # Normalize if needed
image_array = np.expand_dims(image_array, axis=0)
# Use the float values (features) as needed
# Assuming features is a list of 4 float values
input_names = [input.name for input in ort_session.get_inputs()]
features_array = np.array([day_sin, day_cos, month_sin, month_cos]).astype(np.float32).reshape(1, -1)
onnx_inputs = {input_names[0]: image_array, input_names[1]: features_array}
# Process the image and features to make a prediction
onnx_output = ort_session.run(None, onnx_inputs)
local_high, local_low = onnx_output[0][0][0], onnx_output[0][0][1]
local_high = celsius_to_fahrenheit(local_high)
local_low = celsius_to_fahrenheit(local_low)

print(f"local high {inference_high} local low {inference_low}")
#sc = SeldonClient(microservice_endpoint=endpoint)
#client_prediction = sc.microservice(
#    json_data=batch, method="predict"
#)
#print(client_prediction)

