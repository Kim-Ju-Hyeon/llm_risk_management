import os
import pandas as pd


# Todo: This function just load the dataset but I think we need to make additional Custom Dataset for batch computation -> 
def make_dataset(root_dir, ag_news_filename='ag_news.csv', automotive_news_filename='automotive_news.csv'):
    ag_news_filepath = os.path.join(root_dir, 'data', ag_news_filename)
    automotive_news_filepath = os.path.join(root_dir, 'data', automotive_news_filename)
    
    ag_news = pd.read_csv(ag_news_filepath, index_col=0)
    automotive_news = pd.read_csv(automotive_news_filepath, index_col=0)
    
    input_text = pd.concat([ag_news, automotive_news])
    input_text = input_text.reset_index(drop=True)
    
    return input_text