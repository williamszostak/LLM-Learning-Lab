from dotenv import load_dotenv
import os
import pathlib
import json
from openai import OpenAI
import pandas
import numpy
from ast import literal_eval


def main():

    init()

    embeddings = read_csv_to_dataframe(csv_file_path=config['vector_file'])
    question = get_user_question()
    best_matches = find_most_similar_content(text=question, embeddings=embeddings, max=3)
    prompt = generate_prompt(question=question, document_sections=best_matches)

    print("\n------------------------------------")
    print("Here's the prompt we'll send to GPT:")
    print("------------------------------------\n")
    print(prompt['content'])
    print("\n------------------------------------\n")


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

    vector_dir = os.path.join(script_dir, 'data', 'vectors')
    vector_file = os.path.join(vector_dir, 'ka-pow_vectors.csv')

    prompt_dir = os.path.join(script_dir, 'prompts')
    user_prompt_file = os.path.join(prompt_dir, 'rag_user_prompt.txt')

    config['openai_api_key'] = openai_api_key
    config['script_dir'] = script_dir
    config['html_dir'] = html_dir
    config['vector_file'] = vector_file
    config['user_prompt_file'] = user_prompt_file


def read_csv_to_dataframe(csv_file_path: str) -> pandas.DataFrame:
    """
    Reads a CSV file into a pandas DataFrame and processes embedding data into numpy arrays.

    This function takes the path to a CSV file as input and reads the content into a pandas DataFrame. The function
    assumes that there is an 'Embedding' column in the CSV which contains string representations of lists. It converts
    these string representations into actual Python lists using `literal_eval` and then transforms them into numpy arrays
    for numerical operations.

    Parameters:
        csv_file_path (str): The path to the CSV file to be read.

    Returns:
        pandas.DataFrame: A DataFrame containing the data from the CSV file with the 'Embedding' column converted to numpy arrays.
    """
    df = pandas.read_csv(csv_file_path, index_col=None)
    df['Embedding'] = df.Embedding.apply(literal_eval).apply(numpy.array)
    return df


def get_user_question() -> str:
    """
    Prompts the user to enter a question and returns the question as a string.

    Returns:
        str: The text input by the user.
    """
    print("\nEnter a question about Ka-Pow! Comics Cafe")
    print("And we'll generate a prompt that includes relevant information")
    print("to help GPT answer the question..\n")

    return input("Question: ")


def find_most_similar_content(text: str, embeddings: pandas.DataFrame, max: int) -> pandas.DataFrame:
    """
    Finds and returns the most similar content from a DataFrame of embeddings based on the cosine similarity
    to a given text.

    This function computes the embedding for a given piece of text using the `get_gpt_embedding_vector` function,
    then calculates the cosine similarity of this embedding against all embeddings in the provided DataFrame.
    It sorts these embeddings by similarity in descending order, selects the top `max` entries, and returns these
    entries without their embedding vectors.

    Parameters:
        text (str): The text to find similar content for.
        embeddings (pandas.DataFrame): A DataFrame containing the embeddings and other related information.
                                       The DataFrame must have a column named 'Embedding' which contains
                                       numpy arrays of embeddings.
        max (int): The maximum number of similar entries to return.

    Returns:
        pandas.DataFrame: A DataFrame of the most similar content, sorted by similarity, containing the same
                          columns as the input DataFrame minus the 'Embedding' column.

    """
    text_embedding = get_gpt_embedding_vector(text=text)
    embeddings['Similarity'] = embeddings.Embedding.apply(
        lambda e: cosine_similarity(text_embedding, e)
    )
    most_similar = (
        embeddings.sort_values('Similarity', ascending=False)
        .head(max)
        .drop(['Embedding'], axis=1)
        .reset_index()
    )
    return most_similar


def generate_prompt(question: str, document_sections: pandas.DataFrame) -> str:
    prompt_template = read_file(config['user_prompt_file'])
    
    section_text = ''
    for index, row in document_sections.iterrows():
        section_text += '<section>\n'
        section_text += f"Page: {row.Page}\n"
        section_text += f"Section: {row.Section}\n"
        section_text += f"Content:\n{row.Content}\n"
        section_text += '</section>\n'

    prompt = prompt_template.format(web_extracts=section_text, question=question)

    prompt_message = {
        "role": "user", 
        "content": prompt
        }
    return prompt_message


def print_best_matches(best_matches: pandas.DataFrame) -> None:
    """
    Prints the details of the best matching documents or content sections found.

    Parameters:
        best_matches (pandas.DataFrame): A DataFrame that includes at least the columns 'Similarity', 'Page',
                                         'Section', and 'Content'. Each row represents a match with these details.

    """
    print("\n--------------------------------")
    for index, row in best_matches.iterrows():
        print(f"\n#{index+1} match: Similarity: {row.Similarity:.4f}")
        print(f"Page: {row.Page}")
        print(f"Section: {row.Section}")
        print(f"Content:\n{row.Content}\n")
    print("--------------------------------")


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
    return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))


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


if __name__ == '__main__':
    main()
