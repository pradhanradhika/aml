'''
import requests

PLAID_CLIENT_ID = "67d1adf7859c7e0025980e09"
PLAID_SECRET = "20a92ad8f600d08cc8e17ba94dc75d"

url = "https://sandbox.plaid.com/sandbox/public_token/create"

payload = {
    "client_id": PLAID_CLIENT_ID,
    "secret": PLAID_SECRET,
    "institution_id": "ins_109508",  # Replace with the bank's institution ID
    "initial_products": ["transactions"]
}

response = requests.post(url, json=payload)
data = response.json()

PUBLIC_TOKEN = data.get("public_token")
print("Generated Public Token:", PUBLIC_TOKEN)
'''


import requests
url = "https://sandbox.plaid.com/item/public_token/exchange"

PLAID_CLIENT_ID = "67d1adf7859c7e0025980e09"
PLAID_SECRET = "20a92ad8f600d08cc8e17ba94dc75d"
PUBLIC_TOKEN = "public-sandbox-3dbd8654-9084-4bb8-b531-c97cec36809c"


payload = {
    "client_id": PLAID_CLIENT_ID,
    "secret": PLAID_SECRET,
    "public_token": PUBLIC_TOKEN 
}

response = requests.post(url, json=payload)
data = response.json()

ACCESS_TOKEN = data.get("access_token")
print("Generated Access Token:", ACCESS_TOKEN)
