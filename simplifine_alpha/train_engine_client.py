'''
    Simplfine is an easy-to-use, open-source library for fine-tuning LLMs models quickly on your own hardware or cloud.
    Copyright (C) 2024  Simplifine Corp.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
import requests
import json
from .url_class import _url

def send_train_query(query:dict={}):
    """
    Send a training query to the server endpoint.

    This function sends a POST request with the provided query data to a predefined
    server endpoint and returns the server's JSON response. If an error occurs during
    the request, it returns an error message.

    Parameters:
    -----------
    query : dict, optional
        The query data to be sent to the server. Default is an empty dictionary.

    Returns:
    --------
    response_data : dict
        The JSON response from the server, if the request is successful.
        In case of an error, a dictionary containing the error message.

    Raises:
    -------
    requests.exceptions.RequestException
        If an error occurs during the HTTP request, such as connection issues or
        invalid responses.

    Examples:
    ---------
    Example usage of the function:

    >>> query_data = {"param1": "value1", "param2": "value2"}
    >>> response = send_train_query(query_data)
    >>> print(response)
    {'result': 'success', 'data': {...}}

    Notes:
    ------
    - Ensure the server is running and accessible at the specified URL.
    - The headers are set to indicate JSON data.
    """

    url = _url
    headers = {'Content-Type': 'application/json'}  # Set the headers to indicate JSON data
    payload = query # Prepare the payload with the data list
    print('sending stuff', payload)
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        
        # Return the JSON response from the server
        return response.json()
    
    except requests.exceptions.RequestException as e:
        # Handle any exceptions that occur during the request
        return {"error": str(e)}
    
def get_company_status(api_key:str=''):
    """
    Retrieve the company status from the server endpoint.

    This function sends a POST request with the provided API key to a predefined
    server endpoint and returns the company's status as a JSON response. If an
    error occurs during the request, it returns an error message.

    Parameters:
    -----------
    api_key : str, optional
        The API key to authenticate the request. Default is an empty string.

    Returns:
    --------
    response_data : dict
        The 'response' field from the server's JSON response, if the request is successful.
        In case of an error, a dictionary containing the error message.

    Raises:
    -------
    requests.exceptions.RequestException
        If an error occurs during the HTTP request, such as connection issues or
        invalid responses.

    Examples:
    ---------
    Example usage of the function:

    >>> api_key = "your_api_key_here"
    >>> status = get_company_status(api_key)
    >>> print(status)
    {'status': 'active', 'details': {...}}

    Notes:
    ------
    - Ensure the server is running and accessible at the specified URL.
    - The headers are set to indicate JSON data.
    - The 'response' field is extracted from the server's JSON response.
    """
    
    url = _url
    headers = {'Content-Type': 'application/json'}  # Set the headers to indicate JSON data
    payload = {'api_key':api_key} # Prepare the payload with the data list

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        
        # Return the JSON response from the server
        return response.json()['response']
    
    except requests.exceptions.RequestException as e:
        # Handle any exceptions that occur during the request
        return {"error": str(e)}

# TESTS
if __name__ == '__main__':
    query = {'content': 'hello', 'api_key': 'OmidNLPIE', 'job_name': 'test', 'type': 'clm', 'job_id': '1234'}
    res = send_train_query(query)
    print(res)