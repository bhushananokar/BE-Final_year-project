import requests

url = "http://localhost:8000/predict"
symptoms = "Frequent urination,Increased thirst,Unexplained weight loss,Fatigue , Blurred vision ,Slow-healing sores"
response = requests.post(url, json={"symptoms": symptoms})
print(response.text)
