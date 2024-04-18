from dotenv import load_dotenv
import os
import pathlib
import json
from openai import OpenAI
from reprlib import repr
import numpy as np


def main():
    """
    Main function that initializes the environment, collects three pieces of text from the user,
    computes their embeddings using OpenAI's GPT model, and compares the embeddings using cosine similarity.

    This function begins by initializing necessary settings or configurations. It then prompts the user to input
    three separate pieces of text. For each text, the function retrieves an embedding from OpenAI's GPT model and
    stores it. After all embeddings are retrieved, the function calculates and displays the cosine similarity
    between consecutive pairs of embeddings, including the similarity between the last and the first embedding.

    Outputs:
        - Prints the embedding vectors of each text input.
        - Prints the cosine similarity between each pair of consecutive embeddings.
    """
    init()

    embeddings = []
    max = 3

    print(f"\nEnter {max} pieces of text and we'll compare their embedding vectors.\n")

    for i in range(max):
        text = input(f"Text {i+1}: ")
        embedding_object = get_gpt_embedding(text=text)
        embeddings.append(embedding_object.data[0].embedding)
        print(f"\nEmbedding vector: {repr(embeddings[i])}\n")
    
    for a in range(max):
        if a == max-1:
            b = 0
        else:
            b = a + 1
        s = cosine_similarity(embeddings[a], embeddings[b])
        print(f"Cosine similarty between Text {a+1} and Text {b+1}: {s}")
    print()


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


def cosine_similarity(a, b):
    """
    Calculates the cosine similarity between two vectors.

    Cosine similarity is a measure of similarity between two non-zero vectors
    that measures the cosine of the angle between them.

    Parameters:
        a (array_like): First input vector.
        b (array_like): Second input vector.

    Returns:
        float: Cosine similarity between vectors a and b.

    Raises:
        ValueError: If either `a` or `b` is a zero vector (as the norm will be zero, leading to division by zero).
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == '__main__':
    main()
