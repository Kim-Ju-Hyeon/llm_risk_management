import click
import traceback
import os
import datetime
import pytz
from easydict import EasyDict as edict
import yaml
import pandas as pd

from utils.train_helper import mkdir
from utils.logger import setup_logging

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

@click.command()
@click.option('--conf_file_path', type=click.STRING, default=None)
def main(conf_file_path):
    config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))
    now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
    date = now.strftime('%m%d_%H%M%S')
    log_name = 'log_'+ date
    
    log_file = os.path.join(config.root_dir, 'log', log_name)
    logger = setup_logging('INFO', log_file, logger_name=str(date))
    logger.info(f"Writing log file to {log_file}")
    logger.info(f"Exp instance id = {date}")
    
    try:
        if torch.cuda.is_available() and config.device == 'gpu':
            torch_dtype = torch.float16
        else:
            config.device = 'cpu'
            torch_dtype = torch.float32
            
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        pipeline = transformers.pipeline(
            config.task,
            model=config.model_name,
            tokenizer=tokenizer,
            torch_dtype=torch_dtype,
            device=config.device
        )

        df = pd.DataFrame(columns=["Category", "Content"])
        
        for topic in config.topics:
            category = topic['category']
            keyword = topic['keyword']
            
            x = f"You are a journalist responsible for writing articles. Imagine you specialize in a {category}, and you're asked to write an article related to that {keyword}. Please proceed to write an article that fits the given {keyword}."
            
            sequences = pipeline(
                    x,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    max_length=max_length,
            )
            
            df = df.append({"Category": category, "Content": sequences[0]["generated_text"].replace(input_text, "")}, ignore_index=True)

        csv_path = os.path.join(config.root_dir, config.save_dir)
        df.to_csv(csv_path, index=False)
        
    except:
        logger.error(traceback.format_exc())
    
    
if __name__ == '__main__':
    main()