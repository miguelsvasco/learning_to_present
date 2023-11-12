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
        "You are a great prompt engineer for GPT4. Your job is to design prompts for GPT4 to create a set of three slides for a fantastic scientific presentation about recent advancements in machine learning. The first slide is always an introduction to the subject, the second slide is always a more in-depth description of recent machine learning methods and the third slide is always a slide with the conclusions."
        "Given the previous prompts, the previous version of the slides, and the associated evaluation score (out of 10) of those slides, you always generate better prompts to improve the slides."
        "Your output is always only the three different prompts for three slides, generated as a python list with three different strings, each one for a prompt. Make sure that the prompts do not include quotation marks or apostrophes. Make sure that the format of the output is correct, with no missing parenthesis and commas between the prompts. Make sure you do not generate any additional text besides the list of prompts."
    ),
)

# Create the evaluator assistant
evaluator_prompt = (
    (
        "You are a great evaluator of slides for scientific presentations about recent advancements in machine learning. "
        "Given a set of three slides as input, you provide a numerical score for the quality of the slides. "
        "Your output is always just the score of the slides, with minimum 0 and maximum 10, generated as a single number. Make sure that your output is just a single number. Make sure you do not generate any additional text."
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


def ask_writer(
    prompt: List[str], images: List[str], score: int, client: OpenAI
) -> List[str]:
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


def ask_evaluator(images: List[str], client: OpenAI) -> int:
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": str(evaluator_prompt),
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
        max_tokens=1,
    )
    print(response.choices[0].message.content)
    score = int(response.choices[0].message.content)
    assert 0 <= score <= 10
    return score


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

    image_paths = []
    for image_file in image_files:
        image_paths.append(os.path.join(folder_path, image_file))

    # Randomly select an image
    return image_paths


def main() -> None:
    # Create the client to OpenAI API
    client = OpenAI()

    # Load initial set of images
    image_folder = "slides/"
    max_iterations = 10
    prompts = [
        "Slide 1/3: Introduction for a scientific presentation about recent developments in machine learning.",
        "Slide 2/3: Recent developments in machine learning.",
        "Slide 3/3: Conclusions and open questions in recent developments in machine learning.",
    ]
    score = 0
    all_scores = [score]
    all_prompts = [prompts]

    # Optimization loop:
    for opt_iteration in range(max_iterations):
        print(f"** Writing prompts for iteration: {opt_iteration}")
        old_image_paths = select_images(
            folder=image_folder, iteration=opt_iteration - 1
        )
        old_slides = []
        for old_image_path in old_image_paths:
            old_slides.append(encode_image(old_image_path))

        # Ask writer for prompts
        new_prompts = ask_writer(
            prompt=prompts, score=score, client=client, images=old_slides
        )
        all_prompts.append(new_prompts)
        print(f"**** New prompts for iteration: {opt_iteration}: \n {new_prompts}")

        # Generate slides with Designer
        for i in range(len(new_prompts)):
            print(f"**** Generating slide #{i} for iteration: {opt_iteration}")
            ask_designer(
                prompt=new_prompts[i],
                client=client,
                iteration=opt_iteration,
                slide_number=i,
                image_folder=image_folder,
            )

        # Ask evaluator for scores
        print(f"**** Evaluating slides for iteration: {opt_iteration}")
        new_image_paths = select_images(folder=image_folder, iteration=opt_iteration)
        new_slides = []
        for new_image_path in new_image_paths:
            new_slides.append(encode_image(new_image_path))
        new_score = ask_evaluator(images=new_slides, client=client)
        all_scores.append(new_score)
        print(f"**** Got score {new_score} for iteration: {opt_iteration}")

        # Update prompts and score
        prompts = new_prompts
        score = new_score

    # Save results
    results_folder = "results/"

    # Save the NumPy array to the file
    np.save(
        os.path.join(results_folder, "all_scores.npy"),
        np.array(all_scores, dtype=np.int32),
    )
    save_list_of_lists_to_file(
        file_path=os.path.join(results_folder, "all_prompts.txt"),
        list_of_lists=all_prompts,
    )


if __name__ == "__main__":
    main()
