import requests
import json
from utils import chunk_text_by_words

def send_data_list(data_list):
    url = 'http://127.0.0.1:5000/embed'  # URL of the server endpoint
    headers = {'Content-Type': 'application/json'}  # Set the headers to indicate JSON data
    payload = {'data_list': data_list}  # Prepare the payload with the data list

    try:
        # Send the POST request to the server
        print(f"Sending POST request to {url} with payload: {payload}")
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        
        # Return the JSON response from the server
        return response.json()
    except requests.exceptions.RequestException as e:
        # Handle any exceptions that occur during the request
        return {"error": str(e)}
    

# function to accept string and chunk it
def send_data_string(string:str='', chunk_length:int=100, overlap:int=10):

    # chunk the string
    data_list = chunk_text_by_words(text=string, chunk_length=chunk_length, overlap=overlap, verbose=False, return_start_end=False)

    url = 'http://127.0.0.1:5000/embed'  # URL of the server endpoint
    headers = {'Content-Type': 'application/json'}  # Set the headers to indicate JSON data
    payload = {'data_list': data_list}  # Prepare the payload with the data list

    try:
        # Send the POST request to the server
        print(f"Sending POST request to {url} with payload: {payload}")
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        
        # Return the JSON response from the server
        return response.json()
    except requests.exceptions.RequestException as e:
        # Handle any exceptions that occur during the request
        return {"error": str(e)}


def send_query(query:str=''):
    url = 'http://127.0.0.1:5000/ask'  # URL of the server endpoint
    headers = {'Content-Type': 'application/json'}  # Set the headers to indicate JSON data
    payload = {'query': query}  # Prepare the payload with the data list

    try:
        # Send the POST request to the server
        print(f"Sending POST request to {url} with payload: {payload}")
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        
        # Return the JSON response from the server
        return response.json()
    except requests.exceptions.RequestException as e:
        # Handle any exceptions that occur during the request
        return {"error": str(e)}


def get_detail():

    """
    Fetches details from the server using the /get_detail endpoint.

    Parameters:
        base_url (str): The base URL of the server (e.g., 'http://localhost:5000').

    Returns:
        dict: The JSON response from the server.
    """
    url = 'http://127.0.0.1:5000/get_detail'  # URL of the server endpoint
    headers = {'Content-Type': 'application/json'}  # Set the headers to indicate JSON data
    try:
        response = requests.post("http://127.0.0.1:5000/get_detail", headers=headers)
        print(response)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx and 5xx)
        
        # Assuming the response is JSON and will be parsed as such
        return response.json()
    except requests.exceptions.RequestException as e:
        # Handle different types of requests exceptions if needed
        print(f"An error occurred: {e}")
        return {"error": "Failed to get index details from server"}




# Example usage
if __name__ == "__main__":
    # data_list = ['a','b']
    # result = send_data_list(data_list)
    # print(result)
    # print(send_query('capital of france'))
    detail = get_detail()
    print(detail)