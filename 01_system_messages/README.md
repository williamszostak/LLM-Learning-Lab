# Code Demo 1: Using System Messages
These two simple scripts illustrate how to connect to the GPT API endpoint, send a prompt, and get the response.

The second script illustrates how to use a **system** prompt to instruct GPT to adopt a persona.

**System** prompts are useful for providing "background" context and setting overall behavior guidelines that GPT should use when responding to any given **user** prompt.

- [1_01_tell_me_a_joke.py](./1_01_tell_me_a_joke.py)
  - Sends GPT a **user** message saying "Tell me a joke."
  - Gets the JSON response from GPT and prints the JSON to the console so you can examine the structure of the response
  - Separately prints response_object['choices'][0]['message']['content'] which is the text of the response that contains the content we're looking for, i.e. a joke
- [1_02_observational_comic.py](./1_02_observational_comic.py)
  - Sends the same **user** message to GPT: "Tell me a joke."
  - But also sends a **system** message instructing GPT to act as an observational comic.
  - Prints the full JSON response and separately prints the text content (i.e. the joke), just like the previous script
  - Note that GPT responds as the system message instructs it to

In both scripts, the prompts (i.e. ```messages```) are hard-coded in the ```main()``` function.

These messages are then passed to the ```get_gpt_response()``` function, which is where the call to the GPT API happens.
