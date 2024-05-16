# SNU ECE Graduate Project

This is a source code of graduate proejct, a research of recommendation system using deep reinforcement learning.


## Prior Researches
Two prior researches are considered.
- _Liwei Huang, Mingsheng Fu, Fan Li_, A deep reinforcement learning based long-term recommender system, _ELSEVIER_, 2020
- _Jin Huang, Harrie Oosterhuis, Bunyamin Cetinkaya_, State Encoders in Reinforcement Learning for Recommendation: A Reproducibility Study, _ACM_, 2022, _https://doi.org/10.1145/3477495.3531716_


## Dataset
[netflix-prize](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data?resource=download), [movielens-25m](https://grouplens.org/datasets/movielens/25m/), [Yahoo R2!](https://webscope.sandbox.yahoo.com/catalog.php?datatype=i&did=67)
Place data at `data/{DATASET_NAME}`.


## Running
See the source code for full descriptions.

__Preprocessing Data__  
Split train/test by 8:2, into smaller files.  
Run `convert_dataset_format.py` at the top directory.
```
python convert_dataset_format.py
```


__Embedding Data Generation__  
Run `generate_embedding_data.py` at the top directory.  
```
# Generate all data
python generate_embedding_data.py

# Generate data of ceratin dataset
python generate_embedding_data.py --dataset movielens_25m 
```


__Training and Testing__  
Run `main.py` at the top directory.  
```
# train
python main.py --load_folder movielens_25m --save_folder movielens_25m_pg_f --target_model pg --reward_method feedback --dataset movielens_25m --filename train
python main.py --load_folder netflix_prize --save_folder netflix_prize_ac_r --target_model actor_critic --reward_method rating --dataset netflix_prize --filename train

# test
python main.py --load_folder movielens_25m_pg_f --save_folder movielens_25m_pg_f --target_model pg --dataset movielens_25m --filename test --mode test
python main.py --load_folder netflix_prize_ac_r --save_folder netflix_prize_ac_r --target_model actor_critic --dataset netflix_prize --filename test --mode test --reward_method rating
```


## Descriptions (not important)
- \data: modelwise/dataset-wise parameters and raw/processed data  
- \main_network: RL models  
- \running: Control train/test  
- \simulator: RL simulators  
- \trained_models: trained models, and its performances(losses/rewards/hit_rates)  
