import click
import traceback
import os
import datetime
import pytz
from easydict import EasyDict as edict
import yaml
import pandas as pd
from dotenv import load_dotenv

from utils.train_helper import mkdir
from utils.logger import setup_logging

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

@click.command()
@click.option('--conf_file_path', type=click.STRING, default='./config/article_generation.yaml')
def main(conf_file_path):
    load_dotenv()
    huggingface_token = os.getenv("huggingface_token")
    
    config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))
    config.root_dir = os.getcwd()
    
    now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
    date = now.strftime('%m%d_%H%M%S')
    log_name = 'log_'+ date
    
    log_file = os.path.join(config.root_dir, 'logs', log_name)
    logger = setup_logging('INFO', log_file, logger_name=str(date))
    logger.info(f"Writing log file to {log_file}")
    logger.info(f"Exp instance id = {date}")
    
    try:
        if torch.cuda.is_available() and 'cuda' in config.device:
            torch_dtype = torch.float16
        else:
            config.device = 'cpu'
            torch_dtype = torch.float32
            
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, token=huggingface_token)
        
        pipeline = transformers.pipeline(
            config.task,
            model=config.model_name,
            tokenizer=tokenizer,
            torch_dtype=torch_dtype,
            device=config.device
        )
        
        data = []
        for topic in config.topics:
            category = topic['category']
            keyword = topic['keyword']
            
            input_text = f"You are a journalist who writing newspaper article. Imagine you specialize in a {category}, and you're asked to write an article related to that {keyword}. Please proceed to write an article that fits the given {keyword}. The article you write must adhere to the standard format of articles and conclude as a well-structured piece with a clear beginning, middle, and end."
            
            sequences = pipeline(
                    input_text,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    max_length=config.max_length,
            )
            
            data.append({"Category": category, "Keyword": keyword, "Content": sequences[0]["generated_text"].replace(input_text, "")})

        df = pd.DataFrame(data)
        csv_path = os.path.join(config.root_dir, config.save_dir)
        mkdir(csv_path)
        df.to_csv(os.path.join(csv_path, "articles.csv"), index=False)
    except:
        logger.error(traceback.format_exc())
    
    
if __name__ == '__main__':
    main()