import os
import json
from autogen import AssistantAgent, UserProxyAgent
import autogen
import replicate
import requests
from datetime import datetime
from constant import openai_key


def config_list_from_json(filename):
    # Load the content from the JSON file
    with open(filename, 'r') as file:
        config_content = file.read()

    # Replace reference with the actual value from the constant
    config_content = config_content.replace(".constant.openai_key", openai_key)

    # Parse the content back to a list
    config_list = json.loads(config_content)

    return config_list


def img_review(img, input):
    output = replicate.run(
        "yorickvp/llava-13b:2facb4a474a0462c15041b78b1ad70952ea46b5ec6ad29583c0b29dbd4249591",
        input={"image": open(img, "rb"),
               "prompt": f"What is happening in the image? From scale 1 to 5 stars, decide how similar the image is to the text prompt {input}?",}
    )

    result = ""
    for item in output:
        result += item
    return result


if __name__ == "__main__":
    config_list = config_list_from_json('./config/config.json')
    # Check if api_key has been correctly substituted
    api_key_value = config_list[0].get('api_key')
    if not api_key_value or api_key_value == ".constant.openai_key":
        print("Error: openai_key was not correctly loaded!")
        # Handle error (e.g., exit the app, raise an exception, etc.)
    else:
        print("openai_key has been correctly loaded!")

    img1 = "imgs/IMG_0143.JPG"
    print(img_review(img1, "Dog in the sofa"))
