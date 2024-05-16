from utils import get_path

import numpy as np
import pandas as pd






## movielens 25M
def load_movielens_25M(filename: str, use_timestamp: bool) -> pd.DataFrame:
    """
    Returns
    -------
    ratings: user_id, item_id, rating, timestamp
    (not implemented yet) movies: item_id, title, genres
    """

    # rating file
    ratings = pd.read_csv(get_path(f'data/movielens_25m/{filename}.csv'), index_col=False)
    ratings = ratings.rename(columns={'movie_id': 'item_id'})
    ratings = ratings.reset_index(drop=True)
    ratings = ratings.astype({'user_id': np.int32, 'item_id': np.int32, 'rating': np.int32, 'timestamp': str})
    if use_timestamp:
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
        ratings = ratings.sort_values(by='timestamp')

    # # movie_id, title, genres
    # movies = pd.read_csv(f'{folder_path}/movielens_25m/movies.csv')
    # movies = movies[['movieId', 'title', 'genres']]
    # movies = movies.rename(columns={'movieId': 'item_id'})

    return ratings


## netflix prize
def load_netflix_prize(filename: str, use_timestamp: bool) -> pd.DataFrame:
    """
    Returns
    -------
    ratings: user_id, item_id, rating, timestamp
    """

    ratings = pd.read_csv(get_path(f'data/netflix_prize/{filename}.csv'), index_col=False)
    ratings = ratings.astype({'user_id': np.int32, 'item_id': np.int32, 'rating': np.int32, 'timestamp': str})

    if use_timestamp:
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], format='%Y-%m-%d')
        ratings = ratings.sort_values(by='timestamp')

    return ratings


## yahoo R2
def load_yahoo_r2(filename: str) -> pd.DataFrame:
    """
    Parameters
    ----------
    filename: str
        'test_0'~'test_9', 'train_0'~'train_9': certain file
        'train': all train files (DEPRECATED)
        'test': all test files (DEPRECATED)
        None: all train and test files (DEPRECATED)
    
    Returns
    -------
    two `pd.Dataframes` if `filename == None`, one `pd.Dataframe` otherwise
    each dataframes: user_id, item_id, rating
    """

    ratings = pd.read_csv(get_path(f'data/yahoo_r2/{filename}.csv'), index_col=False)
    ratings = ratings.astype({'user_id': np.int32, 'item_id': np.int32, 'rating': np.int32})

    return ratings
