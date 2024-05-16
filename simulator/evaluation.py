from data.dataloader import (
    load_movielens_25M, load_netflix_prize, load_yahoo_r2
)

import torch
import numpy as np
import pandas as pd




class EvaluationSimulator:
    def __init__(self, dataset_params: dict, device: torch.device):
        """
        This simulator is for time-independent recommendation system.
        That is, `timestamp` column is not cosidered when generating the action.

        Args
        ----------
        self.ratings: dict[np.int32, pd.DataFrame]
            key: user_id, value: rating data of each users
            'timestamp' column is not used in this simulator.
        self.user_ids: chosen users for the episode
            this is NOT an index of self.users
        """
        self.device = device

        # env setting
        self.action_type = dataset_params['action_type']
        self.user_ids: np.ndarray[np.int32]    # target users, 1-d array
        self.state: np.ndarray[np.float32]    # (len(user_ids), 2, self.episode_length)

        self.N = dataset_params['top_n']

        # load ratings file
        self.ratings: pd.DataFrame
        if dataset_params['rating'] == 'netflix_prize':
            ratings = load_netflix_prize(dataset_params['rating_file'], use_timestamp=False)

        elif dataset_params['rating'] == 'movielens_25m':
            ratings = load_movielens_25M(dataset_params['rating_file'], use_timestamp=False)
        
        elif dataset_params['rating'] == 'yahoo_r2':
            ratings = load_yahoo_r2(dataset_params['rating_file'])
        
        else:
            print(f"Unsupported rating file type: {dataset_params['rating']}")
            exit(1)
        ratings = ratings.loc[:, ['user_id', 'item_id', 'rating']]
        print("Rating data is loaded.")

        # parameters
        self.users = np.unique(ratings.to_numpy()[:,0])
        self.items = np.unique(ratings.to_numpy()[:,1])
        self.num_total_items = dataset_params['num_items']

        # self.epsilon_min = dataset_params['epsilon']
        # self.epsilon_decay_step = dataset_params['epsilon_decay_step']
        # self.epsilon = 0.5 if self.epsilon_min != 0 else 0

        # tools
        self.valid_item_shelve: torch.FloatTensor
        self.progress_shelve: torch.BoolTensor
        self.ratings = ratings.set_index(['user_id', 'item_id'])

        print("Simulator is ready.")
    

    def reset(self, user_ids: np.ndarray[np.int32]) -> np.ndarray[np.int32]:
        """
        Initial case: `state(prev_action)` is set to 0
        """
        self.user_ids = user_ids
        
        valid_item_shelve = np.zeros((len(user_ids), self.num_total_items), dtype=np.float32)
        progress_shelve = np.ones((len(user_ids),), dtype=np.bool_)
        random_initial_state = np.zeros((len(user_ids),), dtype=np.float32)
        for i , id in enumerate(user_ids):
            target_items = self.ratings.loc[id, :].index.to_numpy()
            valid_item_shelve[i, target_items] = 1
            random_initial_state[i] = np.random.choice(target_items)

        self.valid_item_shelve = torch.FloatTensor(valid_item_shelve).to(self.device)
        self.valid_item_shelve[
            np.arange(len(user_ids)),
            random_initial_state.astype(np.int32)
        ] = 0
        self.progress_shelve = torch.logical_and(
            torch.BoolTensor(progress_shelve).to(self.device),
            self.valid_item_shelve.sum(dim=1) > 0
        )

        return random_initial_state
    

    def step(self, action_likelihood: torch.FloatTensor) \
        -> tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor]:
        """
        TODO: epsilon-greedy action selection is not implemented yet. modify if needed

        Parameters
        ----------
        action_likelihood: shape (len(user_ids), total items in dataset)
            Tensor at `self.device`  \\
            likelihood of each action, result of policy network  \\
            ALWAYS positive, high -> more probable

        Returns
        -------
        actions: shape (len(user_ids),)
            chosen action for each user  \\
            at cpu
        
        feedback_reward: shape (len(user_ids),)
            1: top-N recommendation contains the chosen item, 0: otherwise  \\
            at cpu
            
        rating_reward: shape (len(user_ids),)
            rating score of the chosen item  \\
            at cpu

        in_progress: shape (len(user_ids),)
            whether each episode is in progress or not  \\
            at `self.device`
        """

        if self.action_type == 'deterministic':
            actions = torch.argmax(action_likelihood*self.valid_item_shelve, dim=1)
        elif self.action_type == 'stochastic':
            actions = torch.zeros_like(
                torch.Tensor(action_likelihood.shape[0],), device=self.device, dtype=torch.int64
            )
            actions[self.progress_shelve] = torch.multinomial(
                action_likelihood[self.progress_shelve]*self.valid_item_shelve[self.progress_shelve], 1
            ).squeeze(1)

        top_n_index = torch.topk(action_likelihood, self.N, dim=1)[1]
        feedback_reward = torch.where(
            self.valid_item_shelve[
                torch.arange(self.valid_item_shelve.shape[0]).reshape(-1,1),
                top_n_index
            ].sum(axis=1) > 0,
            1.0,
            0.0
        ).to(device='cpu')

        rating_reward = torch.zeros(
            (self.valid_item_shelve.shape[0],), dtype=torch.float32
        )
        rating_reward[self.progress_shelve] = torch.tensor(
            self.ratings.loc[
                zip(
                    self.user_ids[self.progress_shelve.cpu().numpy()],
                    actions[self.progress_shelve].cpu().numpy()
                ),
                'rating'
            ].to_numpy(),
            dtype=torch.float32
        )

        self.valid_item_shelve[
            torch.arange(len(self.user_ids), device=self.device)[self.progress_shelve],
            actions[self.progress_shelve]
        ] = 0

        in_progress = self.progress_shelve.clone()
        self.progress_shelve = torch.logical_and(
            self.progress_shelve, self.valid_item_shelve.sum(dim=1) > 0
        )

        return actions, feedback_reward, rating_reward, in_progress