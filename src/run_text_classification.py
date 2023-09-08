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
from dataset.article_dataset import make_dataset
from models.llm import LLM


@click.command()
@click.option('--conf_file_path', type=click.STRING, default='./config/text_classification_config.yaml')
def main(conf_file_path):
    now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
    date = now.strftime('%m%d_%H%M%S')
    
    # Add get your own Huggingface Token and Open AI API Key and add it in .env
    load_dotenv()
    huggingface_token = os.getenv("huggingface_token")
    openai_key = os.getenv("open_ai_key")

    # all hyperparameters and configurations are in config directory
    config = load_yaml(conf_file_path)

    # add keys in config yaml file temporally for LLM class
    config.huggingface_token = huggingface_token
    config.openai_key = openai_key

    # The prompts that used in this project should managed in prompt directory
    prompt = load_yaml(os.path.join(config.root_dir, 'prompt', 'document_classification_prompt.yaml'))
    input_text = make_dataset(config.root_dir)

    # load the LLM class that we want
    llm = LLM(config)

    new_df = pd.DataFrame(columns=['Text', 'Automotive_domain', 'model_description_1', 'Hyundai_group', 'model_description_2', 'Mobis', 'model_description_3'])
    
    for index, row in input_text.iterrows():
        # cleaning the input article 
        article = re.sub(r'[^\w\s]', '', row.text).strip()
    
        bool_output, cleaned_output = llm.evaluate_text(article, prompt.first_step)

        new_row = pd.DataFrame({'text': article, 'Automotive_domain': bool_output, 'model_description_1': cleaned_output})
        new_df = pd.concat([new_df, new_row])

    automotive_text = result_df[new_df['Automotive_domain'] == True]
    for index, row in automotive_text.iterrows():
        bool_output, cleaned_output = llm.evaluate_text(row['Text'], prompt.second_step)

        new_df.at[index, 'Hyundai_group'] = bool_output
        new_df.at[index, 'model_description_2'] = cleaned_output
        
    
    save_path = os.path.join(config.root_dir, 'exp')
    mkdir(save_path)
    new_df.to_csv(os.path.join(save_path, f'classified_article_{date}.csv', index=False))
        


if __name__ == '__main__':
    main()
