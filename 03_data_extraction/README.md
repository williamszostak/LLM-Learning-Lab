# Code Demo 3: Data Extraction
This script demonstrates how to use the GPT API to extract data from text and format it as JSON.

This can then be used programmatically to send the data to an API, load it in a database for analysis, etc.

The premise is that an auto insurance company logs customer calls using voice-to-text technology.

We will parse the call transcript and extract data about an auto accident that the policy-holder has reported during the call.

We will format the data as JSON, specifying the field names, so it can be loaded into the claims system via API.

- [call_transcript_1.txt](./data/call_transcript_1.txt) contains the text of the call from the cutomer
- [extract_claim_system_prompt.txt](./prompts/extract_claim_system_prompt.txt) contains the **system** prompt
  - This contains instructions for GPT for the data we are looking to extract
  - Includes field names for the JSON attributes along with descriptions of what these fields contain
  - Notice the JSON includes nested attributes
  - Instructs GPT what to do if it can't find values for some of the fields
  - Tells GPT what format the user prompt will be in (transcript delimited by triple-quotes)
- [3_01_extract_claim_info.py](./prompts/extract_claim_user_prompt.txt) contains a template for the **user** prompt
  - This is a simple shell which the call transcript will be inserted into
  - The prompt template uses the delimiters that are specified in the system prompt
- [3_01_extract_claim_info.py](3_01_extract_claim_info.py)
  - This script reads the transcript and plugs it into the **user** prompt, and sends it to GPT along with the **system** prompt
  - It prints the resulting JSON to the console
  - Note that instead of calling the API endpoint directly, this script leverages the ```openai``` python package.
    - ```client = OpenAI()``` instantiates a client to call the API
    - ```client.chat.completions.create()``` calls the chat completions endpoint with the prompts and waits for the response
    - This library abstracts away the mechanics of calling the API directly
    - It also provides easy mechanisms for:
      - asynchronous responses
      - streaming responses