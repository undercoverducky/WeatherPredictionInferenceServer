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

def retrieve_sat_img(date):
    austin_bbox = (-98.2, 29.85, -97.4, 30.7)
    gibs_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    wms = WebMapService(gibs_url)
    layer = "MODIS_Terra_CorrectedReflectance_TrueColor"
    img_format = "image/jpeg"  # Format of the image to retrieve
    img_size = (512, 512)  # Size of the image (width, height)


    response = wms.getmap(layers=[layer],
                              styles=[''],
                              srs='EPSG:4326',
                              bbox=austin_bbox,
                              size=img_size,
                              format=img_format,
                              time=date.strftime("%Y-%m-%d"))

    encoded_image = base64.b64encode(response.read()).decode('utf-8')
    image = Image.open(io.BytesIO(base64.b64decode(encoded_image)))
    return image
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

def load_model(date=None):
    local_model_path = f'Model-{date}.onnx'
    session = ort.InferenceSession(local_model_path)
    return session

def test_local_onnx_file(date):
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
def test_local_server_endpoint():
    date = datetime.now() - timedelta(days=10)
    image = retrieve_sat_img(date)
    data = fetch_data(date)
    image_array = np.array(image)
    print(image_array.shape)
    print(data)
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
    print(content)
    celsius_to_fahrenheit(daily_high_temp)
    inference_high = celsius_to_fahrenheit(content["data"]["ndarray"][0])
    inference_low = celsius_to_fahrenheit(content['data']["ndarray"][1])

    print(f"endpoint high {inference_high} endpoint low {inference_low}")


def test_local_sequence_server_endpoint():
    seq_len = 20
    image_list_sequence = []
    ground_truth_highs = []
    ground_truth_lows = []

    days_collected = 0
    date = datetime.now() - timedelta(days=10)
    print("Collecting past 20 days of data...")
    while days_collected < seq_len:
        image = retrieve_sat_img(date)
        if image is not None:  # Check if image is available for the date
            data = fetch_data(date)
            image_array = np.array(image)
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

            ground_truth_highs.append(celsius_to_fahrenheit(daily_high_temp))
            ground_truth_lows.append(celsius_to_fahrenheit(daily_low_temp))

            day_sin = math.sin(2 * math.pi * date.day / 31)
            day_cos = math.cos(2 * math.pi * date.day / 31)
            month_sin = math.sin(2 * math.pi * date.month / 12)
            month_cos = math.cos(2 * math.pi * date.month / 12)
            date_features = [day_sin, day_cos, month_sin, month_cos]

            image_list.extend(date_features)
            image_list_sequence.extend(image_list)

            days_collected += 1

        # Move back one day
        date -= timedelta(days=1)
    print(f"Finished collecting {len(image_list_sequence)} length input data")
    # Prepare batch for the new endpoint
    batch = {"data": {"ndarray": image_list_sequence}}

    endpoint = "http://0.0.0.0:9000/api/v1.0/predictions"
    headers = {'Content-Type': 'application/json'}

    # Make the prediction by sending a POST request
    response = requests.post(endpoint, headers=headers, data=json.dumps(batch))
    content = json.loads(response.content.decode('utf-8'))
    print(content)
    # Assuming the response contains an array of high and low temperatures for each date
    for i in range(seq_len):
        inference_high = celsius_to_fahrenheit(content["data"]["ndarray"][i][0])
        inference_low = celsius_to_fahrenheit(content["data"]["ndarray"][i][1])
        print(f"Date {i + 1}:")
        print(f"Ground truth high: {ground_truth_highs[i]}, Inference high: {inference_high}")
        print(f"Ground truth low: {ground_truth_lows[i]}, Inference low: {inference_low}")


test_local_sequence_server_endpoint()

