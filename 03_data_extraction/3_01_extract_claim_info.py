from dotenv import load_dotenv
import os
import pathlib
import json
from openai import OpenAI


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
    prompt_file = os.path.join(config['script_dir'], 'prompts', 'extract_claim_system_prompt.txt')
    prompt = read_file(prompt_file)
    prompt_message = {
        "role": "system", 
        "content": prompt
        }
    return prompt_message


def get_user_prompt():
    """
    Constructs a user prompt message by integrating a transcript text into a predefined prompt template.

    This function reads a prompt template from a text file located in a subdirectory 'prompts' within the directory
    specified by 'config['script_dir']'. It also reads a transcript file from a subdirectory 'data' within the same directory.
    The function then formats the prompt template with the content of the transcript file to create a customized user prompt.
    The prompt is packaged into a dictionary representing the message with the role set to 'user' and the content set to the
    formatted prompt.

    Returns:
        dict: A dictionary containing the user role and the formatted prompt content.

    Example of returned dictionary:
        {
            "role": "user",
            "content": "Based on your conversation about [topic], it seems like ..."
        }
    """
    prompt_file = os.path.join(config['script_dir'], 'prompts', 'extract_claim_user_prompt.txt')
    prompt_template = read_file(prompt_file)

    transcript_file = os.path.join(config['script_dir'], 'data', 'call_transcript_1.txt')
    transcript_info = read_file(transcript_file)

    prompt = prompt_template.format(transcript_text=transcript_info)

    prompt_message = {
        "role": "user", 
        "content": prompt
        }
    return prompt_message


def get_gpt_response(messages):
    """
    Sends a list of messages to a GPT model endpoint for completion and returns the response.
    Uses the OpenAI python library to call the API and get the response.
    See OpenAI's python library documentation for more info:
    https://github.com/openai/openai-python

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
    client = OpenAI(api_key=config['openai_api_key'])

    response = client.chat.completions.create(
        model=config['model'],
        messages=messages,
        temperature=config['temperature']
    )
    return response


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
    
    response = response_object.choices[0].message.content
    print(f'\nResponse message content:\n\n{response}\n')


if __name__ == '__main__':
    main()
