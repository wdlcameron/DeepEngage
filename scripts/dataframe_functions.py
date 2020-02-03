from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import random
import urllib
import numpy as np
from pathlib import Path
import pandas as pd
from functools import partial
import re
import datetime
import imageio


"""
General Functions:  These functions are primarily concerned with handling the dataframe that has been assembled
"""
def load_dataframe(file_name = 'output.csv'):
    return pd.read_csv(file_name)


def get_post_id(url): return url.split(r'/')[-2]

def unique_features(df, feature):
    return df[feature].unique()

def avg_likes_per_user(df, user_list):
    avg_likes = {user:None for user in user_list if user is not np.nan}

    for user in user_list:  
        avg_likes[user] = df[df['username']==user] ['likes'].mean()
    return avg_likes
        
    
def engagement_metric_avg_likes(df):
    avg_likes = avg_likes_per_user(df, unique_features(df, 'username'))
    #print(avg_likes)
    if 'engagement_factor_avg_likes' not in df.columns: df['engagement_factor_avg_likes'] = np.nan
    for (index, _) in df.iterrows():
        username = df.at[index, 'username']
        likes = df.at[index, 'likes']
        df.at[index, 'engagement_factor_avg_likes'] = likes/avg_likes[username]
        
    return df


def engagement_factor_rolling_mean(df):
    df['rolling_avg'] = np.nan
    for username in df['username'].unique():
        sorted_df = df[df['username'] == username].sort_values(by = ['posttime'], inplace = False).reset_index()

        window = 20
        num_rows = len(sorted_df)
        if num_rows <window + 2: sorted_df['rolling_avg'] = sorted_df['likes'].mean()
        else:
            sorted_df['rolling_avg'] = sorted_df.rolling(window, min_periods = window-4)['likes'].mean()
            sorted_df['rolling_avg'].loc[0:window-2] = sorted_df.at[window-1,'rolling_avg']
            sorted_df['rolling_avg'].loc[num_rows - window:] = sorted_df.at[num_rows-window -1,'rolling_avg']
        for (index, row) in sorted_df.iterrows():
            #print (username, index, sorted_df.at[index, 'index'], sorted_df.at[index, 'Rolling Avg'])
            df.at[sorted_df.at[index, 'index'], 'rolling_avg'] = sorted_df.at[index, 'rolling_avg']
            
    df['engagement_factor_moving_avg'] = df['likes']/df['rolling_avg']            
    return df

def export_df(df, output_name = 'output_df.csv'):
    df.to_csv(output_name)
    
        
def read_timestamp(timestamp):
        dt = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    
        return  [('date'    , dt.day),
                 ('hour'    , dt.hour),
                 ('minute'  , dt.minute),
                 ('second'  , dt.second),
                 ('month'   , dt.month),
                 ('year'    , dt.year),
                 ('day_name', dt.isoweekday())]
        
def parse_datetimes(df):

    for r in df.itertuples():
        index = r.Index
        timestamp = r.posttime
        if isinstance(timestamp, str) and timestamp:
            parameters = read_timestamp(timestamp)
            for label, value in parameters: df.at[index, label] = value
                
    return df

def create_performance_windows(df, factor, window = 0.1):
    def _inner_window(i):
        if np.isnan(i): return ""
        elif i>1+window: return 'High'
        elif 1+window>=i>=1-window: return 'Normal'
        elif i<1-window: return 'Low'
        else: return 'Unknown'
        
    df['performance'] = [_inner_window(i) for i in df[factor]]
    return df
                


def extract_colour_information(file_str, output_folder = 'Output', ext = '.jpg'):
    folder_path = Path(output_folder)
    filename =  file_str + ext
    if (folder_path/filename).exists():
        img = imageio.imread(folder_path/filename)

        def _inner_calc_range(img, i):
            pct = np.percentile(img[:,:,i], (5,95))
            return pct[1] - pct[0]

        _pct_calc = partial(_inner_calc_range, img)

        colours = ['red','green','blue']
        colour_values = {colour:img[:,:,i].mean() for i, colour in enumerate(colours)}
        colour_values['brightness'] = img.mean()
        hist_values = {colour+'_range': _pct_calc(i) for i, colour in enumerate(colours)}
        hist_values['contrast'] = _pct_calc([range(0,3)])

        colour_values.update(hist_values)
        return colour_values

    else:
        return {}


def fill_in_colour_information(df, output_folder = 'Output'):
    for (index, row) in df.iterrows():
        post_url = df.at[index, 'Links']
        image_name = get_post_id(post_url)

        output_folder = Path(output_folder)
        colour_info = extract_colour_information(image_name, output_folder)
        for param, value in colour_info.items():
            df.at[index, param] = value
    return df



def post_processing_single(df, metric):
    """
    df = engagement_metric_avg_likes(df)
    df = engagement_factor_rolling_mean(df)
    df = parse_datetimes(df)
    df = create_performance_windows(df, 'engagement_factor_moving_avg')
    df = fill_in_colour_information(df)
    """
    df['filename'] = [url.split(r'/')[-2] for url in df['Links']]

    
    df = metric(df)
    
    for name in df.columns:
        if name.startswith('Unnamed'): df.drop(name, axis = 1, inplace = True)
    
    export_df(df, 'Post_Processed.csv')
    
    return df

    
        
def post_processing(df):
    df['filename'] = [url.split(r'/')[-2] for url in df['Links']]

    
    df = engagement_metric_avg_likes(df)
    df = engagement_factor_rolling_mean(df)
    df = parse_datetimes(df)
    df = create_performance_windows(df, 'engagement_factor_moving_avg')
    df = fill_in_colour_information(df)
    
    for name in df.columns:
        if name.startswith('Unnamed'): df.drop(name, axis = 1, inplace = True)
    
    export_df(df, 'Post_Processed.csv')
    
    return df


if __name__ == '__main__':
    print('Testing not implemented just yet... stay tuned')