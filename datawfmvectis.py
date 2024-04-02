import requests
import json
from ratelimit import limits, sleep_and_retry

# Set the base URL for the Warframe Market API
base_url = "https://api.warframe.market/v1"

# Set the endpoint for searching riven auctions
endpoint = "/auctions/search"

# Set the headers for the request
headers = {
    "Platform": "pc",
    "Accept": "application/json"
}

# Define the parameter set for vectis rivens
params = {
    "type": "riven",
    #"weapon_url_name": "vectis",
    "sort_by": "price_asc",
    'mastery_rank_min' : "8",
}

# Define the rate limit (3 requests per second)
@sleep_and_retry
@limits(calls=3, period=1)
def make_request(url, params, headers):
    response = requests.get(url, params=params, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Request failed with status code: {response.status_code}")
    return response.json()

# Load existing data from the JSON file if it exists
try:
    with open("vectis_rivens_data.json", "r") as file:
        existing_data = json.load(file)
except FileNotFoundError:
    existing_data = {"payload": {"auctions": []}}

# Make the GET request to the API
new_data = make_request(base_url + endpoint, params, headers)

# Combine the existing data with the new data
combined_data = existing_data["payload"]["auctions"] + new_data["payload"]["auctions"]

# Remove duplicates based on the "id" field
unique_data = list({auction["id"]: auction for auction in combined_data}.values())

# Create the updated data structure
updated_data = {"payload": {"auctions": unique_data}}

# Save the updated data to the JSON file
with open("vectis_rivens_data.json", "w") as file:
    json.dump(updated_data, file, indent=4)

print("Data appended to vectis_rivens_data.json and duplicates removed")