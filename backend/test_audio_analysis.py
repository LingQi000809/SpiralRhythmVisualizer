import requests

url = "http://127.0.0.1:8000/process-audio"

with open("/Users/ling/Downloads/supernatural_opening.mp3", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print("Status:", response.status_code)
print(response.json())
