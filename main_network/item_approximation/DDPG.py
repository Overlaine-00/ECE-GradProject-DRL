from main_network.embedding import ItemEmbeddingModel
from simulator.feedback_based import FeedbackSimulator
from main_network.NN.GRU import GRURecurrent
from utils import get_path

import numpy as np
import torch
from torch.functional import F
from torch.distributions import Normal




class DDPG:
    def __init__(self, model_params: dict, device: torch.device):
        assert model_params['action_type'] == 'deterministic', \
            f"Only deterministic policy is supported in DDPG, whereas {model_params['action_type']} is entered."
        assert model_params['action_space'] == 'continuous', \
            f"Only continuous action space is supported in DDPG, whereas {model_params['action_space']} is entered."
        assert model_params['use_embedding'], \
            "Item approximation requires item embedding model."
        
        self.device = device

        self.gamma = model_params['gamma']
        self.rnn_size = model_params['rnn_state_dim']

        self.item_embedding = ItemEmbeddingModel(
            model_params, device
        )
        self.state_dimension = model_params['state_embedding_dim']

        self.actor = GRURecurrent(
            self.state_dimension, 2*self.state_dimension, self.rnn_size, model_params['num_rnn_layer']
        ).to(self.device)
        self.critic = GRURecurrent(
            self.state_dimension, 1, self.rnn_size, model_params['num_rnn_layer']
        ).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=model_params['learning_rate']
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=model_params['learning_rate']
        )
    

    def choose_action(self, state: torch.FloatTensor) -> torch.FloatTensor:
        hidden_state, params = self.actor(state)
        mean, std = torch.chunk(params, 2, dim=-1)
        action_dist = Normal(0, 1).sample(size=mean.size()).to(self.device)
        action_dist = action_dist*(F.softplus(std) + 1e-5) + mean
        return hidden_state, action_dist
    

    def train(
        self, simulator: FeedbackSimulator, user_ids: np.ndarray[np.int32]
    ):
        episode_reward_Q = torch.zeros(
            len(user_ids), dtype=torch.float32
        )
        episode_hit_rate = torch.zeros(
            (len(user_ids),), dtype=torch.float32
        )
        episode_critic_loss = torch.FloatTensor([0])
        episode_length = torch.zeros(
            (len(user_ids),), dtype=torch.float32
        )

        state: torch.FloatTensor = torch.FloatTensor(
            simulator.reset(user_ids)
        ).to(self.device)
        hidden_state: torch.FloatTensor = torch.zeros(
            (len(user_ids), self.rnn_size), dtype=torch.float32
        ).to(self.device)

        state = self.item_embedding.get_item_embedding(state.to(torch.int64))
        value: torch.FloatTensor = self.critic(state, hidden_state.detach())[1]

        for _ in range(30):
            # step
            hidden_state, action_vector = self.choose_action(state)
            hidden_state = hidden_state.detach()
            chosen_action, feedback_reward, in_progress = simulator.step(action_vector)

            state = self.item_embedding.get_item_embedding(chosen_action.to(torch.int64))

            # calculate reward 1
            behavior_V = value.detach()

            # calculate reward 2
            value = self.critic(state, hidden_state.detach())[1].squeeze(1)
            feedback_reward = 1.2*feedback_reward - 0.2    # f_t -> V_t by 0 -> -0.2, 1 -> 1

            target_V = feedback_reward + self.gamma*value
            
            episode_reward_Q[in_progress] += (
                feedback_reward + self.gamma*value.detach()    # self.gamma*pres_value corresponds to value of target, so .detach() is added
            )[in_progress]

            # record hit rate
            in_progress = in_progress.cpu()
            episode_hit_rate[in_progress] += feedback_reward.cpu()[in_progress]
            episode_length[in_progress] += 1
            
            if not torch.any(in_progress):
                break


        reward_Q_as_loss = torch.mean(-episode_reward_Q)
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        reward_Q_as_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        episode_reward_Q = episode_reward_Q.detach().cpu()
        episode_length = torch.where(episode_length == 0, 1, episode_length)
        return -episode_reward_Q.mean(), episode_reward_Q, episode_hit_rate/episode_length, True

    
    def save_model(self, folder: str):
        torch.save(self.actor.state_dict(), get_path(f"{folder}/actor.pt"))
        torch.save(self.critic.state_dict(), get_path(f"{folder}/critic.pt"))
        self.item_embedding.save_model(folder)


    def load_model(self, path: str):
        if not self.item_embedding.load_model(path):
            print("The program will be terminated.")
            exit()
        try:
            self.actor.load_state_dict(torch.load(get_path(f"{path}/actor_critic/actor.pth")))
            self.critic.load_state_dict(torch.load(get_path(f"{path}/actor_critic/critic.pth")))
            return True
        except FileNotFoundError:
            print(f"Model file not found in {path}. Train a new model.")
            return False