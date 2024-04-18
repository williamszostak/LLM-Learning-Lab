from dotenv import load_dotenv
import os
import pathlib
import json
from openai import OpenAI
from pprint import pprint

def main():
    """
    Main execution function that initializes the environment, processes user input, and prints the embedding details.

    This function starts by calling `init()` to set up necessary configurations or initialize the environment.
    It then captures user input through `get_user_input()` and passes this text to `get_gpt_embedding(text)`,
    which generates an embedding vector using a GPT model for the input text. The embedding vector is
    then printed using `print_embedding(embedding_object)`, and the length of the embedding vector is also displayed.

    Outputs:
        Prints the embedding vector and its length to the standard output.
    """
    init()

    text = get_user_input()

    embedding_object = get_gpt_embedding(text=text)

    print_embedding(embedding_object=embedding_object)
    print(f"Length of embedding vector: {len(embedding_object.data[0].embedding)}\n")


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


def get_user_input() -> str:
    """
    Prompts the user to enter some text and returns the text as a string.

    Returns:
        str: The text input by the user.
    """
    print("\nEnter some text and we'll return an embedding of the text.\n")
    return input("Text: ")


def get_gpt_embedding(text: str) -> object:
    """
    Generates an embedding vector for the given text using OpenAI's GPT model specified in the configuration.

    This function initializes a connection to the OpenAI API using an API key retrieved from the configuration.
    It sends a request to the API to generate an embedding for the provided text using the model also specified
    in the configuration. The function assumes that `config` is a dictionary with keys for 'openai_api_key' and
    'embedding_model'.

    Parameters:
        text (str): The text for which to generate the embedding.

    Returns:
        object: The response object from OpenAI API which contains the embedding data.

    Example:
        Assuming 'config' is properly set with an OpenAI API key and a valid embedding model, if we pass:
        text = "example text"
        The function will return an object containing the embedding vector for "example text".
    """
    client = OpenAI(api_key=config['openai_api_key'])

    response = client.embeddings.create(
        model=config['embedding_model'],
        input=text
    )
    return response


def print_embedding(embedding_object: object):
    """
    Prints a simplified version of the embedding vector from the given embedding object.

    This function extracts the embedding vector from the provided embedding object, which is expected
    to be a response object from the OpenAI API. 
    It then pretty-prints the response object with a truncated version of the embedding vector 
    for readability, displaying the structure and contents of the response.

    Parameters:
        embedding_object (object): The embedding object from which to extract and print the embedding vector.
                                   This object is expected to have a method `model_dump()` that returns
                                   the embedding data in JSON format.
    """
    embedding_json = embedding_object.model_dump()
    vector = embedding_json['data'][0]['embedding']
    vector_abbrev = vector[:5] + ['...'] + vector[-1:]
    embedding_json['data'][0]['embedding'] = vector_abbrev
    print("\nResponse from OpenAI embedding API:\n")
    pprint(embedding_json)
    print()


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
