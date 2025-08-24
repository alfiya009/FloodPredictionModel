import requests
import json

def test_api():
    url = "http://127.0.0.1:5000/predict_flood"
    data = {"city": "Mumbai"}
    
    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
