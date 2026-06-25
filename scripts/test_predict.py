import urllib.request
import json
import urllib.error

url = 'http://127.0.0.1:5000/predict'
payload = {
    'credit_score': 410,
    'geography': 'Karnataka',
    'gender': 'Female',
    'age': 52,
    'tenure': 1,
    'balance': 14500000.0,
    'num_products': 3,
    'has_credit_card': 0,
    'is_active_member': 0,
    'estimated_salary': 17500000.0
}

req = urllib.request.Request(
    url, 
    data=json.dumps(payload).encode('utf-8'), 
    headers={'Content-Type': 'application/json'},
    method='POST'
)

try:
    response = urllib.request.urlopen(req)
    print("Success response:")
    print(response.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print(f"Error response code: {e.code}")
    print(e.read().decode('utf-8'))
