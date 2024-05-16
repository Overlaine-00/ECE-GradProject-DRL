from utils import get_path

import os
from typing import Union
from tqdm import tqdm
import re
import json

import numpy as np
import pandas as pd


## movielens 25m
def convert_movielens_25m_to_csv(
    rating: bool = False
):
    folder_path = os.getcwd()

    if rating:
        ratings = pd.read_csv(f'{folder_path}/movielens_25m/ratings.csv')
        ratings.rename(columns={'userId': 'user_id', 'movieId': 'movie_id', 'rating': 'rating'}, inplace=True)
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

        item_ids = ratings['movie_id'].unique(); item_ids.sort()
        ratings['movie_id'] = ratings['movie_id'].map(lambda x: np.where(item_ids == x)[0][0])
        np.save(f'{folder_path}/movielens_25m/rating_item_id_index.npy', item_ids)

        user_ids = ratings['user_id'].unique()
        user_ids = np.random.permutation(user_ids)
        sample_size = len(user_ids)//5+1
        
        ratings = ratings.set_index('user_id', drop=True)
        for i in range(5):
            sample = user_ids[i*sample_size:(i+1)*sample_size]
            sample_ratings = ratings.loc[sample, :].reset_index(drop=False)

            if i < 4:
                sample_ratings.to_csv(get_path(f'{folder_path}/movielens_25m/train_{i}.csv', index=False))
            else:
                sample_ratings.to_csv(get_path(f'{folder_path}/movielens_25m/test_{i-4}.csv', index=False))


## netflix prize
def convert_netflix_prize_to_csv(
    rating: bool = False
):
    folder_path = os.getcwd()

    if rating:
        print("Converting netflix prize RATING data to csv...")
        ratings = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])
        
        item_ids = ratings['item_id'].unique(); item_ids.sort()
        ratings['item_id'] = ratings['item_id'].map(lambda x: np.where(item_ids == x)[0][0])
        np.save(f'{folder_path}/netflix_prize/rating_item_id_index.npy', item_ids)

        for i in range(1, 5):
            file_path = get_path(f'{folder_path}/netflix_prize/combined_data_{i}.txt')

            data_lines = []
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    if ':' in line:
                        movie_id = int(line[:-2])
                    else:
                        line = line[:-1]    # remove '\n'
                        user_id, rating, timestamp = line.split(',')
                        data_lines.append([int(user_id), movie_id, int(rating), timestamp])
            
            ratings = pd.concat([
                ratings,
                pd.DataFrame(data_lines, columns=['user_id', 'item_id', 'rating', 'timestamp'])
            ])
        
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], format='%Y-%m-%d')
        ratings.sort_values(by='user_id', inplace=True)

        user_ids = ratings['user_id'].unique()
        user_ids = np.random.permutation(user_ids)
        sample_size = len(user_ids)//10+1

        ratings = ratings.set_index('user_id', drop=True)
        for i in range(10):
            sample = user_ids[i*sample_size:(i+1)*sample_size]
            sample_ratings = ratings.loc[sample, :].reset_index(drop=False)


            if i < 8:
                sample_ratings.to_csv(get_path(f'{folder_path}/netflix_prize/train_{i}.csv', index=False))
            else:
                sample_ratings.to_csv(get_path(f'{folder_path}/netflix_prize/test_{i-8}.csv', index=False))


## yahoo R2
def convert_yahoo_to_csv(filename: str):
    folder_path = os.getcwd()

    print(f"Converting Yahoo R2 {filename} rating data to csv...")
    file_path = get_path(f'{folder_path}/yahoo_r2/{filename}.txt')

    with open(file_path, 'r') as f:
        lines = f.readlines()
        data_lines = []
        for line in tqdm(lines):
            elements = re.split('\t', line[:-1])
            data_lines.append(elements)

    ratings = pd.DataFrame(data_lines, columns=['user_id', 'item_id', 'rating'])
    ratings = ratings.astype({'user_id': np.int32, 'item_id': np.int32, 'rating': np.int32})
    
    ratings.to_csv(get_path(f'{folder_path}/yahoo_r2/{filename}.csv'), index=False)




if __name__ == '__main__':
    convert_movielens_25m_to_csv(rating=True)
    convert_netflix_prize_to_csv(rating=True)
    for i in range(10):
        convert_yahoo_to_csv(f'train_{i}')
        convert_yahoo_to_csv(f'test_{i}')