from dotenv import load_dotenv
import os
import pathlib
import json
import requests
from pprint import pprint


def main():
    """
    Executes the main routine of the program.
    
    This function initializes the program, sends a message to a GPT model endpoint
    to generate a response to a predefined message, and prints the response message
    content.

    Example:
        This function can be called to initiate the program. It sends a message
        to the GPT model endpoint asking for a joke, retrieves the response, and
        prints the generated joke.
    """
    init()

    messages =  [{
        "role": "user", 
        "content": "Tell me a joke."
        }]
    
    response_object = get_gpt_response(messages=messages)
    
    print(f'\nAPI response object:')
    pprint(response_object)

    response = response_object['choices'][0]['message']['content']
    print(f'\nResponse message content:\n{response}\n')


def init():
    """
    Initialize the configuration for the application.

    This function sets up the configuration by loading environment variables from a .env file,
    retrieving the OpenAI API key, and updating the configuration dictionary accordingly.

    Note:
        This function assumes that the application's configuration file is named 'config.json'
        and resides in the parent directory of the script file.

    Raises:
        FileNotFoundError: If the .env file or the configuration file ('config.json') is not found.
    """
    global config

    parent_dir = pathlib.Path(__file__).parent.parent.resolve()
    env_file_path = os.path.join(parent_dir, '.env')
    load_dotenv(env_file_path)
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    config_file_path = os.path.join(parent_dir, 'config.json')
    config = get_config(config_file_path)

    config['openai_api_key'] = openai_api_key
    config['api_headers'] = {
        'accept': '*/*',
        'Authorization': f'Bearer {openai_api_key}',
    }


def get_gpt_response(messages):
    """
    Sends a list of messages to a GPT model endpoint for completion and returns the response.

    Args:
        messages (list): A list of dictionaries representing the messages to be completed.
                         Each dictionary contains a "role" and a "content".
                         Example: [{"role": "system", "content": "You are a helpful assistant"},
                                   {"role": "user", "content": "What is generative AI?"}]
                         See OpenAI's API documentation for more info:
                         https://platform.openai.com/docs/api-reference/introduction

    Returns:
        dict: A dictionary containing the response from the GPT model endpoint.

    Raises:
        HTTPError: If the HTTP request to the GPT model endpoint fails.
        KeyError: If the response JSON does not contain expected keys.
    """
    request_body = {
        'model': config['chat_model'],
        'messages': messages,
        'temperature': config['temperature']
    }
    
    print(f"\nCalling endpoint: {config['chat_endpoint']}")
    print('with messages:')
    pprint(messages)

    response = requests.post(
        url=config['chat_endpoint'],
        headers=config['api_headers'],
        json=request_body
    )
    json = response.json()

    if response.ok:
        return json
    else:
        print(f'\nReceived response status code: {response.status_code}')
        if (json['error']['message']):
            print(f"Error message: {json['error']['message']}")
            response.raise_for_status()


def get_config(file_path: str) -> dict:
    """
    Reads a JSON configuration file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the JSON configuration file.

    Returns:
        dict: A dictionary containing the contents of the JSON configuration file.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        JSONDecodeError: If the contents of the file are not valid JSON.

    Example:
        If 'config.json' contains {"key": "value"}, calling get_config('config.json')
        will return {'key': 'value'}.
    """
    with open(file_path, 'r') as config_file:
        config = json.loads(config_file.read())
    return config


if __name__ == '__main__':
    main()
