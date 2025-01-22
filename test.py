import os.path

import requests

path = os.path.join("Keys", "OpenRouter.txt")
if os.path.exists(path):
    with open(path, "r") as file:
        api_key = file.read()
    # Define the API endpoint
    url = "https://openrouter.ai/api/v1/auth/key"
    # Set up the headers with the Authorization bearer token
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    try:
        # Make the GET request to the API endpoint
        response = requests.get(url, headers=headers)
        # Raise an exception for HTTP error responses
        response.raise_for_status()
        # Parse the JSON response
        data = response.json()
        # Check if there's an error in the response data
        if "error" in data:
            print(f"Error: {data['error']['message']}")
        else:
            # Print the API key details.
            for key, value in data.items():
                print(f"{key}: {value}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except ValueError:
        print("Error: Unable to parse JSON response.")
