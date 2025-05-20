import requests

url = "https://ac8a-2401-4900-2080-68f9-e98f-1672-2460-5744.ngrok-free.app/predict"
files = {'file': open(r"C:\Users\sathya sai\OneDrive\Desktop\Projects\TBP_project\Plant_Disease_Detection\test\test\AppleCedarRust1.JPG", 'rb')}

response = requests.post(url, files=files)

print("Status code:", response.status_code)
print("Response text:", response.text)

try:
    print("JSON:", response.json())
except Exception as e:
    print("Failed to decode JSON:", e)
