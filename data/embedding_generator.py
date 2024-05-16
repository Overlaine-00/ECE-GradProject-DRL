from data.dataloader import (
    load_movielens_25M, load_netflix_prize, load_yahoo_r2
)
from utils import get_path

import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd




class EmbeddingGenerator:
    def __init__(self, dataset_name: str, file_name: str):
        """
        Args
        ----------
        dataset_name: str
            one of 'movielens_25m', 'netflix_prize', 'yahoo_r2'
        file_name: str
            file name of the dataset
        """
        self.dataset_name = dataset_name
        self.file_name = file_name
        self.ratings: pd.DataFrame
        if self.dataset_name == 'netflix_prize':
            ratings = load_netflix_prize(self.file_name, use_timestamp=False)

        elif self.dataset_name == 'movielens_25m':
            ratings = load_movielens_25M(self.file_name, use_timestamp=False)
        
        elif self.dataset_name == 'yahoo_r2':
            ratings = load_yahoo_r2(self.file_name)

        else:
            print(f"Unsupported rating file type: {self.dataset_name}")
            exit(1)

        ratings = ratings.loc[:, ['user_id', 'item_id', 'rating']]
        self.ratings = ratings.set_index(['user_id', 'item_id'])
        
        self.users = np.unique(ratings.to_numpy()[:,0])
    

    def generate_data(
        self, data_type: str, user_ids: np.ndarray[np.int32], context_size: int, save: bool=True
    ) -> tuple[np.ndarray[np.int64], np.ndarray[np.int64]]:
        max_len = max([len(self.ratings.loc[id, :]) for id in user_ids])
        raw_array_iter = iter(self.ratings.loc[id, :].index.to_numpy() for id in user_ids)
        padded_items = np.stack([
            np.pad(arr, (0, max_len-len(arr)), 'constant', constant_values=-1) 
            for arr in raw_array_iter
        ], axis=0)

        sliding_items = sliding_window_view(
            padded_items, (1, context_size+1)
        ).reshape(-1, context_size+1)
        sliding_items = sliding_items[sliding_items[:, -1] != -1]

        if data_type == 'cbow':
            target_items, context_items = sliding_items[:, -1], sliding_items[:, :-1]
        elif data_type == 'skip_gram':
            target_items, context_items = sliding_items[:, :-1], sliding_items[:, -1]
        else:
            raise ValueError(f'Invalid data type: {data_type}')
        
        if save:
            if not os.path.exists(get_path(
                f'data/embedding_data/{self.dataset_name}/front_{context_size}'
            )):
                os.makedirs(get_path(
                f'data/embedding_data/{self.dataset_name}/front_{context_size}'
                ))
            try:
                old_target = np.load(get_path(
                f'data/embedding_data/{self.dataset_name}/front_{context_size}/{self.file_name}_{data_type}_target.npy'
                ))
                old_context = np.load(get_path(
                f'data/embedding_data/{self.dataset_name}/front_{context_size}/{self.file_name}_{data_type}_context.npy'
                ))

                target_items = np.concatenate([old_target, target_items])
                context_items = np.concatenate([old_context, context_items])
            except FileNotFoundError:
                pass
            np.save(get_path(
                    f'data/embedding_data/{self.dataset_name}/front_{context_size}/{self.file_name}_{data_type}_target.npy'
                ), target_items
            )
            np.save(get_path(
                    f'data/embedding_data/{self.dataset_name}/front_{context_size}/{self.file_name}_{data_type}_context.npy'
                ), context_items
            )

        return target_items, context_items




def load_embedded_data(
    dataset_name: str, file_name: str, data_type: str, context_size: int
) -> tuple[np.ndarray[np.int64], np.ndarray[np.int64]]:
    '''
    Parameters
    ----------
    data_type: 'cbow' or 'skip_gram'

    Returns
    -------
    target_items, context_items
    '''
    try:
        target_items = np.load(get_path(
            f'data/embedding_data/{dataset_name}/front_{context_size}/{file_name}_{data_type}_target.npy'
        ))
        context_items = np.load(get_path(
            f'data/embedding_data/{dataset_name}/front_{context_size}/{file_name}_{data_type}_context.npy'
        ))
        return target_items, context_items
    
    except FileNotFoundError:
        print(f"Data file not found in {get_path(f'data/embedding_data/{dataset_name}/front_{context_size}')}")
        exit(1)
        return None, None