import os
import pathlib
from langchain_text_splitters import HTMLHeaderTextSplitter


def main():
    """
    Main function that initializes the environment, reads an HTML file, 
    and splits it into self-contained sections based on the header tags
    in the HTML.

    The main function demonstrates how to process and display structured content extracted from an HTML file,
    which could be useful for content analysis or reformatting tasks.

    Outputs:
        - Prints the sequence number, metadata, and content of each HTML section split.
        - Prints a summary message indicating the total number of splits and acknowledging the source HTML file.
    """
    init()

    source_file = config['home_page_path']
    html_splitter = get_html_splitter()
    splits = html_splitter.split_text_from_file(source_file)

    ii=0
    for split in splits:
        ii += 1
        print(f"\nSplit: {ii}")
        print(f"Level:{split.metadata}")
        print(f"Content:\n{split.page_content}")
    
    print(f"\nLangChain identified the headers in {config['home_page']}")
    print("and used these headers to split the file into")
    print(f"{len(splits)} units of self-contained information.")


def init():
    """
    Initializes the global configuration for the script.

    This function sets up a global `config` dictionary that contains paths and settings used throughout the script.
    It calculates the paths based on the location of the script file itself to ensure that paths are relative
    and accurate regardless of the current working directory of the execution environment.

    After execution, the `config` dictionary will include:
    - 'script_dir': The directory of the script.
    - 'html_dir': The directory where HTML files are stored.
    - 'home_page': The filename of the home page HTML file.
    - 'home_page_path': The full path to the home page HTML file.

    Effects:
        Modifies the global `config` dictionary with appropriate paths and file names.
    """

    global config

    script_dir = pathlib.Path(__file__).parent.resolve()
    html_dir = os.path.join(script_dir, 'data', 'source')

    home_page = 'ka-pow.html'
    home_page_path = os.path.join(html_dir, home_page)

    config = {
        'script_dir': script_dir,
        'html_dir': html_dir,
        'home_page': home_page,
        'home_page_path': home_page_path
    }


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
 

if __name__ == '__main__':
    main()
