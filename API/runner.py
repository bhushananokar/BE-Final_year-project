import requests

url = "http://localhost:8000/predict"
symptoms = "fever 101F, continuous cough, fatigue, loss of appetite"
response = requests.post(url, json={"symptoms": symptoms})
print(response.json())