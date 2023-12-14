import click
import traceback
import os
import pandas as pd
from dotenv import load_dotenv
import time
import re
import random
import datetime

from utils.util_fnc import load_yaml, mkdir
from models.llm import LLM


@click.command()
@click.option('--conf_file_path', type=click.STRING, default='./config/text_classification_config.yaml')
def main(conf_file_path):
    # Add get your own Huggingface Token and Open AI API Key and add it in .env
    load_dotenv()
    huggingface_token = os.getenv("huggingface_token")
    openai_key = os.getenv("open_ai_key")

    # all hyperparameters and configurations are in config directory
    config = load_yaml(conf_file_path)
    prompt = load_yaml(os.path.join(config.root_dir, 'prompt', f'{config.task}_prompt.yaml'))

    # load the LLM class that we want
    llm = LLM(config)

    # load the data that we want to classify
    df = pd.read_csv(os.path.join(config.root_dir, 'dataset', f'{config.task}_data.csv'))

    preprocessed_review = []
    translate = []
    product = []
    model_code = []
    keyword = []
    topic = []
    summary = []
    risk_level = []

    for input_text in df['text']:
        # generate the input text
        input_text_dict = generate_input_text(prompt, input_text)

        start_time = time.time()
        elapsed_time = 0

        # Loop to generate and evaluate text until the time limit is reached
        while elapsed_time < self.config.time_limit:
            # Get the generated answer
            gen_text = self.answer(input_text)

            # Check if the generated text is valid
            try:
                eval(cleaned_output)
                preprocessed_review.append(review)

                if config.task == 'voc':
                    translate.append(reply['translate'])
                
                elif config.task == 'onm':
                    product.append(reply['product_name'])
                    model_code.append(reply['model_code'].upper())

                keyword.append(reply['keyword'])
                topic.append(reply['topic'])
                summary.append(reply['summary'])
                risk_level.append(reply['risk_level'])

            except:
                # If the generated text is invalid, continue to the next iteration
                continue

            # Update elapsed time
            elapsed_time = time.time() - start_time
    
    if config.task == 'voc':
        df['translate'] = translate
    elif config.task == 'onm':
        df['product_name'] = product
        df['model_code'] = model_code

    df['keyword'] = keyword
    df['topic'] = topic
    df['summary'] = summary
    df['risk_level'] = risk_level

    save_path = os.path.join(config.save_dir, 'exp')
    mkdir(save_path)
    df.to_csv(os.path.join(save_path, f'summerized_{config.task}.csv', index=False))
        


if __name__ == '__main__':
    main()
