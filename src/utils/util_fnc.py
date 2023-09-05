import os
from os import path
import yaml
import time
import re
import random
from easydict import EasyDict as edict

import numpy as np
import pandas as pd
import torch


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
    """
    Creates a new directory if it does not already exist.
    
    Checks whether the specified folder exists. If it does not,
    the function will create the folder along with any necessary
    intermediate directories.
    
    Parameters:
    folder (str): The path of the directory to be created.
    
    Returns:
    None
    """
    
    # Check if the directory already exists
    if not path.isdir(folder):
        # Create the directory
        # 'makedirs' method will create any necessary intermediate directories
        os.makedirs(folder)



def load_yaml(file_path):
    """
    Load a YAML file and return it as an EasyDict object.
    
    This function reads a YAML file specified by the file_path argument.
    If the file exists, it will be loaded and returned as an EasyDict object.
    If the file does not exist, a FileNotFoundError will be raised.
    
    Parameters:
    file_path (str): The path to the YAML file to be loaded.
    
    Returns:
    EasyDict: An EasyDict object containing the YAML data.
    """
    
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


def cleaning_text(text):
    """
    Cleans the given text.
    
    1. Removes spaces at the beginning and end of the string.
    2. Eliminates all characters except alphabets and spaces.
    
    Parameters:
    text (str): The text to be cleaned.
    
    Returns:
    str: The cleaned text.
    """
    
    text = text.strip()  # Remove spaces at the beginning and end of the string
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove all characters except alphabets and spaces
    return cleaned_text


def generate_input_text(prompt_dict, article):
    """
    Generate input text by randomly choosing a prompt from each category in the provided dictionary.
    
    Parameters:
    prompt_dict (dict): Dictionary containing lists of prompts categorized as role_prompts, step_prompts, example_prompts, and last_prompts.
    article (str): The article text
    
    Returns:
    str: Generated input text
    """
    chosen_role_prompt = random.choice(prompt_dict['role_prompts'])
    chosen_step_prompt = random.choice(prompt_dict['step_prompts'])
    chosen_example_prompt = random.choice(prompt_dict['example_prompts'])
    chosen_last_prompt = random.choice(prompt_dict['last_prompts'])

    input_text_list = [chosen_role_prompt, chosen_step_prompt, chosen_example_prompt, article, chosen_last_prompt]
    # input_text = (f"{chosen_role_prompt}\n"
    #               f"{chosen_step_prompt}\n"
    #               f"{chosen_example_prompt}\n"
    #               f"'''{article}'''\n"
    #               f"{chosen_last_prompt}")
    
    return input_text_list