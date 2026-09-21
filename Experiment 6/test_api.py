import json
import requests

API_URL = "http://127.0.0.1:8000/predict"

with open("sample_input.json", "r") as file:
    employee_data = json.load(file)

response = requests.post(API_URL, json=employee_data)

print("Status Code:", response.status_code)
print("Response:")
print(response.json())

