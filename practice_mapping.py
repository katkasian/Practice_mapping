import warnings
warnings.simplefilter("ignore", UserWarning)

import pandas as pd
import argparse
import re
import os
import time
import winsound


from openai import OpenAI
from dotenv import load_dotenv

# create an openai account and get the API key from here "https://platform.openai.com/account/api-keys"
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def read_from_csv(file_path):
    try:
        # Open the CSV file for reading
        df = pd.read_csv(file_path, index_col=0)
        return df

    except FileNotFoundError:
        print("The specified file was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def get_context_error(err_msg):
    """Search for the time value (assumes it is a single number: seconds)"""
    match = re.search(r'Please try again in (\d.+?)', err_msg)
    if match:
        try:
            sleep_time = re.sub(r'\D', '', match.group(1))
            print(f'Shh! Am trying to sleep for {sleep_time} s')
            return int(sleep_time)
        except ValueError:
            sleep_time(60)
    else:
        raise ValueError("No try again time found in error message")


def call_openai(model, text):
    try:
        response = client.completions.create(
            model=model,
            prompt=text,
            temperature=0,
            max_tokens=10,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.0
        )
        return response
    except Exception as e:
        time.sleep(get_context_error(e.json_body['error']['message']) + 1)


def create_prompt(text, data):
    prompt_text = text + " " + data
    return prompt_text


if __name__ == '__main__':

    ap = argparse.ArgumentParser(description='A script to communicate with OpenAI API to conduct practice mapping')
    ap.add_argument('-p', '--prompt_file', required=True, help='Path to the prompt file')
    ap.add_argument('-d', '--tweet_dataset', required=True, help='Path to the tweet dataset')
    ap.add_argument('-o', '--output_file', required=True, help='Path of the output file')
    args = ap.parse_args()
    file_path = args.prompt_file

    with open(file_path, encoding='utf-8', errors='ignore') as f:
        prompt = f.read()

    dataset = read_from_csv(args.tweet_dataset)
    # shuffling the dataset every time
    validation_dataset = dataset.sample(frac=1)
    output = []

    for data in validation_dataset.iterrows():
        prompt_text = create_prompt(prompt, data[1]['text'])
        result = call_openai("gpt-3.5-turbo-instruct", prompt_text)
        true_label = data[1]['label']
        try:
            predicted_label = result.choices[0].text.strip().split(': ')[1]
        except IndexError:
            predicted_label = result.choices[0].text
        except TypeError:
            predicted_label = 'Not applicable'
        print("true label : {}, predicted label : {} ".format(true_label, predicted_label))
        output_row = [data[1]['text'], predicted_label, true_label]
        output.append(output_row)

    output_df = pd.DataFrame(columns=['tweet_text', 'predicted_label', 'true_label'], data=output)
    output_df.to_csv(args.output_file)
    duration = 1000  # milliseconds
    freq = 2000  # Hz
    winsound.Beep(freq, duration)
