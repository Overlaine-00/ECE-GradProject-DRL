from main_network.embedding import ItemEmbeddingModel
from main_network.feedback_based.actor_critic import ActorCritic
from simulator.feedback_based import FeedbackSimulator
from utils import get_path

import numpy as np
import torch
from torch.functional import F




class PPO:
    def __init__(
        self,
        model_params: dict,
        device: torch.device
    ):
        # parameters
        self.device = device
        self.use_embedding: bool = model_params['use_embedding']

        self.gamma = model_params['gamma']    # reward decay
        self.rnn_size = model_params['rnn_state_dim']

        self.state_dimension: int
        if self.use_embedding:
            self.item_embedding = ItemEmbeddingModel(
                model_params, device
            ).to(self.device)
            self.state_dimension = model_params['state_embedding_dim']
        else:
            self.state_dimension = model_params['num_items']
        
        # memory
        self.memory_size = model_params['memory_size']
        self.memory_counter = 0
        # old state, old prob(1), old value(1), old next state
        self.memory = np.zeros((self.memory_size,), dtype=np.float32)

        # model
        if self.use_embedding:
            self.item_embedding = ItemEmbeddingModel(
                model_params, device
            )
        self.actor_critic = ActorCritic(model_params, device)
        
    
    def run_episode(
        self, simulator: FeedbackSimulator, user_ids: np.ndarray[np.int32]
    ):
        episode_reward_A = torch.zeros(
            (len(user_ids),), dtype=torch.float32, device=self.device
        )