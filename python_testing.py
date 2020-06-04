import json
import urllib.request
# download raw json object
url = "http://127.0.0.1:5000/predict"
data = urllib.request.urlopen(url).read().decode()

# parse json object
obj = json.loads(data)
print(obj)