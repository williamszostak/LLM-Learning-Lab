from dotenv import load_dotenv
import os
import pathlib
import json
import requests


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

    script_dir = pathlib.Path(__file__).parent.resolve()
    parent_dir = script_dir.parent.resolve()
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
    config['script_dir'] = script_dir


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


def read_file(file_path: str) -> str:
    """
    Reads the contents of a text file and returns them as a string.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The contents of the text file as a string.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        IOError: If an error occurs while reading the file.
    """
    try:
        with open(file_path, 'r') as file:
            file_contents = file.read()
        return file_contents
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    except IOError as e:
        raise IOError(f"An error occurred while reading the file '{file_path}': {e}")


def get_system_prompt():
    """
    Retrieves a system prompt from a text file and formats it into a message object.

    This function reads the contents of a text file containing a system prompt,
    creates a message object with the role "system" and the prompt content, and
    returns the message object.

    Returns:
        dict: A message object representing the system prompt, with keys "role" 
        and "content".

    Raises:
        FileNotFoundError: If the system prompt file is not found.
        IOError: If an error occurs while reading the system prompt file.
    """
    prompt_file = os.path.join(config['script_dir'], 'prompts', 'tone_system_prompt.txt')
    prompt = read_file(prompt_file)
    prompt_message = {
        "role": "system", 
        "content": prompt
        }
    return prompt_message


def get_user_prompt():
    """
    Retrieves a user prompt from a text file, customizes it based on user input,
    and formats it into a message object.

    This function reads the contents of a text file containing a user prompt template,
    prompts the user for input to customize the prompt, incorporates the user's input
    into the template, and returns the customized prompt as a message object.

    Returns:
        dict: A message object representing the user prompt, with keys "role" 
        and "content".

    Raises:
        FileNotFoundError: If the prompt file or email file is not found.
        IOError: If an error occurs while reading the prompt file or email file.
    """
    prompt_file = os.path.join(config['script_dir'], 'prompts', 'tone_user_prompt.txt')
    prompt_template = read_file(prompt_file)

    email_file = os.path.join(config['script_dir'], 'data', 'email_message_1.txt')
    email_message = read_file(email_file)

    print(f'\nEmail message:\n{email_message}\n')
    print("Let's adjust the tone of the email.")
    print("For example, make it more professional, casual, direct, or polite.\n")

    tone = input("What is your desired tone? ")

    prompt = prompt_template.format(requested_tone=tone, message_text=email_message)

    prompt_message = {
        "role": "user", 
        "content": prompt
        }
    return prompt_message




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
        'model': config['model'],
        'messages': messages,
        'temperature': config['temperature']
    }
    
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


def main():
    """
    Executes the main routine of the program.
    
    This function initializes the program, sends a message to a GPT model endpoint
    to generate a response to a predefined message, and prints the response message
    content.
    """
    init()

    system_prompt = get_system_prompt()
    user_prompt = get_user_prompt()

    messages =  [
        system_prompt,
        user_prompt
    ]
    
    response_object = get_gpt_response(messages=messages)
    
    response = response_object['choices'][0]['message']['content']
    print(f'\nResponse message content:\n\n{response}\n')


if __name__ == '__main__':
    main()
