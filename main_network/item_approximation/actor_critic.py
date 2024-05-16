from main_network.embedding import ItemEmbeddingModel
from simulator.feedback_based import FeedbackSimulator
from main_network.NN.GRU import GRURecurrent
from utils import get_path

import numpy as np
import torch
from torch.functional import F
from torch.distributions import Normal, TransformedDistribution, AffineTransform




class ActorCritic:
    def __init__(
        self,
        model_params: dict,
        device: torch.device
    ):
        assert model_params['action_type'] == 'stochastic', \
            f"Only stochastic policy is supported in Actor-Critic, whereas {model_params['action_type']} is entered."
        assert model_params['action_space'] == 'continuous', \
            f"This is continuous model. Use discrete model for {model_params['action_space']} action space"
        assert model_params['use_embedding'], \
            "Continuous action space requires item embedding model."
        self.device = device

        self.gamma: float = model_params['gamma']    # reward decay
        self.rnn_size: int = model_params['rnn_state_dim']

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

    
    def choose_action(
        self, state: torch.FloatTensor, hidden_state: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        hidden_state, params = self.actor(state)

        mean, std = torch.chunk(params, 2, dim=-1)
        std = F.softplus(std) + 1e-6
        action_dist = TransformedDistribution(
            Normal(0,1), AffineTransform(mean, std)
        )

        action_sample = action_dist.sample()
        log_prob = action_dist.log_prob(action_sample)
        return hidden_state, action_sample, log_prob
    

    def train(
        self, simulator: FeedbackSimulator, user_ids: np.ndarray[np.int32]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, bool]:
        """
        Returns
        -------
        (loss, reward, hit_rate, done)
        loss: shape (1,)
            average of -reward
        reward: shape (len(user_ids),)
            Q-function in actor-critic
        hit_rate: shape (len(user_ids),)
            corresponds to hit@N
        done: bool
            True if the episode is done (needed to be reset), False otherwise
        """
        episode_reward_Q = torch.zeros(
            (len(user_ids),), dtype=torch.float32
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

        # initial
        state = self.item_embedding.get_item_embedding(state)
        value: torch.FloatTensor = self.critic(state, hidden_state.detach())[1]

        for t in range(50):
            # calculate advantage 1
            episode_reward_Q -= value.detach()

            # step
            hidden_state, action_vector, action_log_prob = self.choose_action(state, hidden_state)
            hidden_state = hidden_state.detach()

            score = self.item_embedding.get_original_item_scores(action_vector)
            chosen_action, feedback_reward, in_progress = simulator.step(score)

            state = self.item_embedding.get_item_embedding(chosen_action.float())
            
            # calculate advantage
            behavior_V = value.detach()
            value = self.critic(state, hidden_state)[1].squeeze(1)
            target_V = feedback_reward + self.gamma*value

            # train actor
            actor_loss = -action_log_prob[torch.arange(len(user_ids)), chosen_action] * target_V.detach()
            actor_loss = torch.sum(actor_loss[in_progress])
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # train critic
            critic_loss = F.mse_loss(
                behavior_V[in_progress],
                target_V[in_progress],
                reduction='sum'
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # recording
            in_progress = in_progress.cpu()
            feedback_reward = feedback_reward.cpu()
            episode_reward_Q[in_progress] *= self.gamma
            episode_reward_Q[in_progress] += feedback_reward[in_progress]
            episode_hit_rate[in_progress] += torch.where(feedback_reward[in_progress] > 0, 1, 0)
            episode_critic_loss += F.mse_loss(behavior_V[in_progress], target_V[in_progress]).detach().cpu()
            episode_length[in_progress] += 1

            if not torch.any(in_progress):
                break

        episode_length = torch.where(episode_length == 0, 1, episode_length)
        episode_reward_Q /= episode_length
        episode_hit_rate /= episode_length
        episode_critic_loss /= torch.max(episode_length)
        return episode_critic_loss, episode_reward_Q, episode_hit_rate, True
    

    def save_model(self, path: str) -> bool:
        self.item_embedding.save_model(path)
        torch.save(self.actor.state_dict(), get_path(f"{path}/actor.pth"))
        torch.save(self.critic.state_dict(), get_path(f"{path}/critic.pth"))
        return True
    

    def load_model(self, path: str) -> bool:
        if not self.item_embedding.load_model(path):
            print("The program will be terminated.")
            exit()
        try:
            self.actor.load_state_dict(torch.load(get_path(f"{path}/actor.pth")))
            self.critic.load_state_dict(torch.load(get_path(f"{path}/critic.pth")))
            return True
        except FileNotFoundError:
            print(f"Model file not found in {path}. Train a new model.")
            return False