from dotenv import load_dotenv
import os
import pathlib
import json
from openai import OpenAI
from langchain_text_splitters import HTMLHeaderTextSplitter
import csv


def main():
    """
    Main execution function that initializes settings, processes multiple HTML files, and outputs results to a CSV file.

    This function performs the following steps:
    1. Initializes global configuration settings.
    2. Retrieves an HTML splitter object configured to split HTML files based on certain headers.
    3. Opens a specified output CSV file and sets up a CSV writer to write data.
    4. Iterates over all HTML files in a specified directory, splits each file into sections, and processes each section:
       a. Constructs a unique identifier for each section based on its metadata.
       b. Generates an embedding vector for the section content using a GPT model.
       c. Prints details about the process to the console.
       d. Writes the file name, section identifier, section content, and embedding vector to the CSV file.

    The function leverages a global configuration (`config`) that includes paths and settings used during processing.
    Outputs of the function are both printed to the console and written to a CSV file for further analysis or storage.

    Outputs:
        - Console: Prints updates about the processing of each section including the file name, section details, and part of the embedding vector.
        - CSV File: Writes the page name, section identifier, section content, and the entire embedding vector for each section processed.
    """

    init()

    html_splitter = get_html_splitter()

    with open(config['output_file'],'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(('Page', 'Section', 'Content', 'Embedding'))

        ii=0
        for source_file in pathlib.Path(config['html_dir']).glob('*.html'):
            splits = html_splitter.split_text_from_file(source_file)
            for split in splits:
                ii += 1
                section = '::'.join(split.metadata.values())
                vector = get_gpt_embedding_vector(f"{section}\n{split.page_content}")
                print(f"\nGot embedding {ii}")
                print(f"File: {source_file.name}")
                print(f"Section: {section}")
                print(f"Vector: {vector[:3] + ['...']}")

                # Write the page content and resulting vector to a CSV file
                writer.writerow((source_file.name, section, split.page_content, vector))


def init():
    """
    Sets up the global configuration by defining paths, loading environment variables, and reading configuration files.

    Global Variables:
        config (dict): A dictionary to hold configuration details which is modified in-place.

    Example:
        Assuming the relevant files and environment variables are set correctly, calling this function
        will populate the `config` dictionary with paths and settings required for other functions to operate properly.
    """
    global config

    script_dir = pathlib.Path(__file__).parent.resolve()
    parent_dir = script_dir.parent.resolve()
    env_file_path = os.path.join(parent_dir, '.env')

    load_dotenv(env_file_path)
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    config_file_path = os.path.join(parent_dir, 'config.json')
    config = get_config(config_file_path)

    html_dir = os.path.join(script_dir, 'data', 'source')

    output_dir = os.path.join(script_dir, 'data', 'vectors')
    output_file = os.path.join(output_dir, 'ka-pow_vectors.csv')

    config['openai_api_key'] = openai_api_key
    config['script_dir'] = script_dir
    config['html_dir'] = html_dir
    config['output_file'] = output_file


def get_html_splitter() -> object:
    """
    Creates and returns an HTMLHeaderTextSplitter object configured to split HTML content based on specified headers.

    Returns:
        object: An instance of HTMLHeaderTextSplitter configured with specific headers to detect and use for splitting HTML content.
    """
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
        ("h4", "Header 4"),
    ]
    return HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
 

def get_gpt_embedding_vector(text: str) -> list:
    """
    Generates an embedding vector for the given text using OpenAI's GPT model specified in the configuration.

    This function initializes a connection to the OpenAI API using an API key retrieved from the configuration.
    It sends a request to the API to generate an embedding for the provided text using the model also specified
    in the configuration. The function assumes that `config` is a dictionary with keys for 'openai_api_key' and
    'embedding_model'.

    Parameters:
        text (str): The text for which to generate the embedding.

    Returns:
        list: The embedding vector from the response object from OpenAI API.

    Example:
        Assuming 'config' is properly set with an OpenAI API key and a valid embedding model, if we pass:
        text = "example text"
        The function will return the embedding vector for "example text".
    """
    client = OpenAI(api_key=config['openai_api_key'])

    response = client.embeddings.create(
        model=config['embedding_model'],
        input=text
    )
    return response.data[0].embedding


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
