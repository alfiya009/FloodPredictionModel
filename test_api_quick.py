import requests
import json

# Test the flood prediction API
url = "http://127.0.0.1:5000/predict_flood"
data = {"city": "Mumbai"}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
