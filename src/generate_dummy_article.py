import click
import traceback
import os
import datetime
import pytz
from easydict import EasyDict as edict
import yaml
import pandas as pd

from utils.train_helper import mkdir, edict2dict
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
        model = config.model_name
        tokenizer = AutoTokenizer.from_pretrained(model)
        
        pipeline = transformers.pipeline(
            config.task,
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device=config.device
        )

    except:
        logger.error(traceback.format_exc())
    
    
    
if __name__ == '__main__':
    main()