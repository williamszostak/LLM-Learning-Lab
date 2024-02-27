# Code Demo 2: Templates and Delimiters
Imagine you're creating a messaging app, or a social media app, or a product review site. 

You want to provide a feature that lets people adjust the tone of their content before posting it.

Maybe they want to make it sound more professional, or more friendly. More to the point, or more flowery. Funnier, or more serious. Or maybe they just want to sound like a pirate! 

This sample script uses GPT to re-write a message in whatever tone you want.

- [2_01_adjust_tone.py](./2_01_adjust_tone.py)
  - Reads a message draft that the user has written but hasn't sent yet: [email_message_1.txt](./data/email_message_1.txt)
  - Prints the draft message to the console
  - Prompts the user to enter a tone
  - The function ```get_system_prompt()``` reads the **system** prompt from a file:
    - It reads the file [tone_system_prompt.txt](./prompts/tone_system_prompt.txt)
    - The system prompt specifies that the message draft will be delimited by ```<message>``` XML tags
    - It also specifies that the requested tone for the re-write will be delimited by ```<tone>``` XML tags
  - The function ```get_user_prompt()``` generates a **user** prompt message in the format that was given by the system prompt
    - It reads the file [tone_user_prompt.txt](./prompts/tone_user_prompt.txt)
    - This file is a template that contains the specified tags, along with placeholders for the content:
      - ```{requested_tone}```
      - ```{message_text}```
    - The function asks the user for the tone
    - Then it plugs the tone and the draft message into the prompt template
  - The function ```get_gpt_response()``` calls the GPT API with the above two prompts
  - The response (the re-written email) is printed to the console




