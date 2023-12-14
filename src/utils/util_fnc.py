import os
from os import path
import yaml
import time
import re
import random
from easydict import EasyDict as edict

import numpy as np
import pandas as pd

from soynlp.normalizer import repeat_normalize


def edict2dict(edict_obj):
    """
    Converts an edict object to a Python dictionary.
    
    This function recursively traverses through the given edict object,
    converting it and its nested edicts into a standard Python dictionary.
    
    Parameters:
    edict_obj (edict): The edict object to be converted.
    
    Returns:
    dict: The converted Python dictionary.
    """
    
    dict_obj = {}
    
    # Iterate through each key-value pair in the edict
    for key, vals in edict_obj.items():
        
        # Check if the value is an edict object
        if isinstance(vals, edict):
            # Recursively convert nested edict to dictionary
            dict_obj[key] = edict2dict(vals)
        else:
            # If value is not an edict, directly assign it to the new dictionary
            dict_obj[key] = vals
    
    return dict_obj


def mkdir(folder):

    # Check if the directory already exists
    if not path.isdir(folder):
        # Create the directory
        # 'makedirs' method will create any necessary intermediate directories
        os.makedirs(folder)


def load_yaml(file_path):
    # Check if the file exists
    if os.path.exists(file_path):
        # Open the file in read mode with utf-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            # Load the YAML file as an EasyDict object
            yaml_file = edict(yaml.load(file, Loader=yaml.FullLoader))
        
        return yaml_file
    else:
        # Raise a FileNotFoundError if the specified file does not exist
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")


def generate_input_text(prompt_dict, text):
    chosen_role_prompt = random.choice(prompt_dict['role_prompt'])
    chosen_step_prompt = random.choice(prompt_dict['step_prompt'])
    chosen_ex_prompt = random.sample(prompt_dict['example_prompt'], len(prompt_dict['example_prompt'])) # example prompt를 더 많이 생성해서 원하는 갯수로 뽑을 수 있게 구현 현재는 example peompt 갯수 자체가 적어서 그냥 다 뽑게 구현
    chosen_fin_prompt = random.choice(prompt_dict['fin_prompt'])


    input_text_dict = {
        'role_prompt': chosen_role_prompt,
        'step_prompt': chosen_step_prompt,
        'chosen_example_prompt': [f"{idx+1}) Review: {ex[0]}, Answer: {ex[1]}" for idx, ex in enumerate(chosen_ex_prompt)],
        'text': preprocess(text),
        'fin_prompt': chosen_fin_prompt
    }
    
    return input_text_dict


punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", 
                "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', 
                "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 
                'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
    

def clean(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])

    for p in punct:
        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
    for s in specials:
        text = text.replace(s, specials[s])

    return text.strip()

def clean_str(text):
    pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)' # E-mail제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # URL제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '<[^>]*>'         # HTML 태그 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '[^\w\s\n]'         # 특수기호제거
    text = re.sub(pattern=pattern, repl='', string=text)
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','', string=text)
    text = re.sub('\n', '.', string=text)

    return text

def normalize(text):
    text = repeat_normalize(text)
    return text


def preprocess(text):
    text = clean(text, punct, punct_mapping)
    text = clean_str(text)
    text = normalize(text)

    return text