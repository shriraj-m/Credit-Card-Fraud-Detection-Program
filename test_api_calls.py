import requests

data = {
    "features": [0.0] * 30,
    "model_type": "xgboost"
}

response = requests.get("http://localhost:8000/")
print(response.json())

prediction_response = requests.post("http://localhost:8000/predict", json=data)
print("\nStatus Code:", prediction_response.status_code)
print("Response Headers:", prediction_response.headers)
print("Response Body:", prediction_response.text)

health_response = requests.get("http://localhost:8000/predict/health")
print("\nStatus Code:", health_response.status_code)
print(f"Response: {health_response.json() if health_response.status_code == 200 else health_response.text}")
