from typing import List
from openai import OpenAI
from tqdm import tqdm
import numpy as np
import os
import ast
import random
import base64
import requests

# Create the writer assistant
writer_context = (
    (
        "You are a great writter of speeches for presentations about recent advancements in machine learning. Given these slides, that make a three-slide presentation about recent advacements in machine learning, write three sentences for each slide."
    ),
)


def save_list_of_lists_to_file(file_path, list_of_lists):
    try:
        with open(file_path, "w") as file:
            for inner_list in list_of_lists:
                for item in inner_list:
                    file.write(item + "\n")
                file.write("\n")  # Add an empty line between inner lists
        print(f"List of lists saved to '{file_path}' successfully.")
    except Exception as e:
        print(f"Error saving list of lists to '{file_path}': {e}")


def string_to_list(input_string):
    try:
        # Safely evaluate the input string using ast.literal_eval
        result = ast.literal_eval(input_string)

        # Check if the result is a list of strings
        if isinstance(result, list) and all(isinstance(item, str) for item in result):
            return result
        else:
            raise ValueError("Input is not a valid list of strings")
    except (SyntaxError, ValueError) as e:
        print(f"Error: {e}")
        return None


# Function to download an image from a URL
def download_image(url: str, filename: str) -> bool:
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open file in binary write mode and write the contents of the response
        with open(filename, "wb") as f:
            f.write(response.content)
        return True
    else:
        return False


def ask_writer(prompt: List[str], score: int, client: OpenAI) -> List[str]:
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": str(writer_context)
                        + f" Your previous prompts were {str(prompt)} and you got a score of {score}. Generate better prompts.",
                    },
                ],
            }
        ],
        max_tokens=3500,
    )
    raw_prompts = response.choices[0].message.content
    print(raw_prompts)
    # Parsing the string into a Python list
    parsed_prompts = string_to_list(raw_prompts)
    assert len(parsed_prompts) == 3
    return parsed_prompts


def ask_designer(
    prompt: str, iteration: int, slide_number: int, client: OpenAI, image_folder: str
) -> None:
    response = client.images.generate(
        model="dall-e-3",
        prompt=str(prompt),
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    folder_path = os.path.join(image_folder, str(iteration))
    filename = os.path.join(folder_path, f"{slide_number}.png")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    assert download_image(image_url, filename=filename)
    return


def ask_writter(images: List[str], client: OpenAI) -> str:
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": str(writer_context),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{images[0]}",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{images[1]}",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{images[2]}",
                        },
                    },
                ],
            }
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content


# Function to encode the image
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def select_images(folder: str, iteration: int) -> List[str]:
    folder_path = os.path.join(folder, str(iteration))

    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Filter out files that are not images
    image_files = [
        f
        for f in files
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
    ]
    assert len(image_files) > 0

    # Randomly select an image
    paths = []
    for file in image_files:
        paths.append(os.path.join(folder_path, file))
    return paths


def main() -> None:
    # Create the client to OpenAI API
    client = OpenAI()

    # Load initial set of images
    image_folder = "slides/"
    iteration = 8

    # Ask writer for speech
    image_paths = select_images(folder=image_folder, iteration=8)
    encoded_images = []
    for image in image_paths:
        encoded_images.append(encode_image(image))
    text = ask_writter(encoded_images, client)
    print(text)


if __name__ == "__main__":
    main()
