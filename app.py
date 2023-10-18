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


def text_image_generation(prompt):
    output = replicate.run(
        "stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316",
        input={"prompt": prompt})
    
    image_url = output[0]
    print(f"Generated image for {prompt}: {image_url}")

    # Download the image
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    shorted_prompt = prompt[:50]
    filename = f"imgs/{shorted_prompt}_{current_time}.png"

    response = requests.get(image_url)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
                file.write(response.content)
        return (f"Image saved to {filename}")
    else:
        return (f"Error: {response.status_code} - {response.text}")

    return output


def img_review(img, input):
    output = replicate.run(
        "yorickvp/llava-13b:2facb4a474a0462c15041b78b1ad70952ea46b5ec6ad29583c0b29dbd4249591",
        input={"image": open(img, "rb"),
               "prompt": f"What is happening in the image? From scale 1 to 5 stars, decide how similar the image is to the text prompt {input}?", }
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

    # img1 = "imgs/IMG_0143.JPG"
    # print(img_review(img1, "Dog in the sofa"))
    # text_image_generation("A dog in the sofa")

    llm_config_assistants = {
        "functions": [
            {
                "name": "text_image_generation",
                "description": "Use the latest text to image generation model to generate image based on a prompt, return the file path of image generated",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to generate image"
                        }
                    },
                    "required": ["prompt"]
                },
            }, {
                "name": "img_review",
                "description": "Use the latest image review model to review the image based on a prompt, return the review result",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "img": {
                            "type": "string",
                            "description": "The path of the image to review with absoluate path"
                        },
                        "input": {
                            "type": "string",
                            "description": "The prompt to review the image"
                        }
                    },
                    "required": ["img", "input"],
                }
            }
        ]
    }

    img_review_agent = AssistantAgent(
        name="expert_text_to_image",
        system_message="You are a text to image AI model expert, you will use text_image_generation function to generate image with prompt provided, and also improve prompt based on feedback provided untill it is 4/5 or 5/5 stars",
        llm_config=llm_config_assistants,
        function_map={
            "img_review": img_review,
            "text_image_generation": text_image_generation
        }
    )

    img_critic_assistant = AssistantAgent(
        name="img_criteria_assistant",
        system_message="You are an image critic, you will use img_review function to review the image with prompt provided, and provide feedback to improve the image",
        llm_config=llm_config_assistants,
        function_map={
            "img_review": img_review,
            "text_image_generation": text_image_generation
        }
    )

    user_proxy_agent = UserProxyAgent(
        name="user_proxy_agent",
        human_input_mode="ALWAYS",
    )

    groupchat = autogen.GroupChat(
        agents=[user_proxy_agent,
                img_review_agent,
                img_critic_assistant],
        messages=[], max_round=50)
    
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config_assistants,
    )

    user_proxy_agent.initiate_chat(
        manager, message="Dog is swimming in the beach"
    )
