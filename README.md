# LLM-Learning-Lab
This content was created for a [Hack.Diversity](https://www.hackdiversity.com/) Learning Lab: 

### "Hack-GPT": LLMs for Software Engineers and Data Analysts

Presented on February 27, 2024

The slides are available in the [presentation](./presentation/) folder.

To get started, first follow the instructions in the [Setup](./README.md#setup) section.

## Demo Code

Follow the instructions in the README files in each of the demo folders to run the code for that demo:

- Demo 1: [Using System Messages](./01_system_messages/README.md)
- Demo 2: [Templates and Delimiters](./02_templates_delimiters/README.md)
- Demo 3: [Data Extraction](./03_data_extraction/README.md)
- Demo 4: [Retrieval-Augmented Generation (RAG)](./04_rag/README.md)

## Setup

### 1. Sign up for an OpenAI API Key
The scripts in this project use the OpenAI API.

You will need to have an OpenAI API key.

You can sign up for API access at [OpenAI](https://openai.com/blog/openai-api)

As of February 2024, new sign-ups receive $5 in free API credits,
and you don't need to provide any payment information to use the free credits.
This may change in the future.

After you sign up, you can create an API key on the [API Keys page](https://platform.openai.com/api-keys)

Make note of the API key when it's created, as it will only be given to you once.
(You can request a new key if you lose it.)

### 2. Update the .env file
Follow the instructions in the [.env_example](./.env_example) file 
to create a .env file and add your API key to it.

### 3. Create and activate a python virtual environment

Open a terminal and cd to the root directory of this project (LLM-Learning-Lab), then run the following commands:

#### Linux or MacOS:
```
python3 -m venv venv/LLM-Lab
source venv/LLM-Lab/bin/activate
```

#### Windows Powershell:
```
python -m venv venv\LLM-Lab
venv\LLM-Lab\bin\activate
```

If successful, you should see (LLM-Lab) preceding each new line in your terminal. This indicates that the virtual environment is active.

You won't need to create the virtual environment again, but you will need to activate it if it's not active.

Tip: If you're using VS Code, you can set the Python interpreter to use your virtual environment by pressing Ctrl-Shift-P and typing Python: Select Interpreter.

### 4. Install required packages in the virtual environment

```pip install -r requirements.txt```

This will install the packages needed by the scripts in this project.

You'll only need to do this once, and they'll be installed into the active virtual environment for you to use now and in the future, any time you activate your virtual environment.

### 5. Run the first script

```
cd 01_system_messages
python 01_01_tell_me_a_joke.py
```

The script should connect to the OpenAI GPT 3.5 Turbo endpoint and ask it to tell a joke. The result should be printed to the console. 

If you have trouble, make sure your API key is correctly set in the .env file, you have internet access, you're not behind a firewall, etc.

### 6. Have fun exploring the scripts!





