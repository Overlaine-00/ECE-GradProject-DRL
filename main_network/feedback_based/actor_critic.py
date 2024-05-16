from main_network.embedding import ItemEmbeddingModel
from simulator.feedback_based import FeedbackSimulator
from simulator.evaluation import EvaluationSimulator
from main_network.NN.GRU import GRURecurrent
from utils import get_path

import numpy as np
import torch
from torch.functional import F




class ActorCritic:
    def __init__(
        self,
        model_params: dict,
        device: torch.device
    ):
        assert model_params['action_type'] == 'stochastic', \
            f"Only stochastic policy is supported in Actor-Critic, whereas {model_params['action_type']} is entered."
        assert model_params['action_space'] == 'discrete', \
            f"This is discrete model. Use continuous model for {model_params['action_space']} action space"
        
        self.device = device
        self.use_input_embedding: bool = model_params['use_embedding']
        self.use_output_embedding: bool = (model_params['action_space'] == 'continuous')

        self.gamma: float = model_params['gamma']
        self.rnn_size: int = model_params['rnn_state_dim']

        self.state_input_dim: int
        if self.use_input_embedding:
            self.item_embedding = ItemEmbeddingModel(
                model_params, device
            )
            self.state_input_dim = model_params['state_embedding_dim']
        else:
            self.state_input_dim = 1
        
        self.actor = GRURecurrent(
            self.state_input_dim, model_params['num_items'], self.rnn_size, model_params['num_rnn_layer']
        ).to(self.device)
        self.critic = GRURecurrent(
            self.state_input_dim, 1, self.rnn_size, model_params['num_rnn_layer']
        ).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=model_params['learning_rate']
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=model_params['learning_rate']
        )
    

    def convert_state(self, state: torch.FloatTensor) -> torch.FloatTensor:
        if self.use_input_embedding:
            return self.item_embedding.get_item_embedding(state.to(torch.int64))
        else:
            return state.unsqueeze(dim=1)
    

    def train(
        self, simulator: FeedbackSimulator, user_ids: np.ndarray[np.int32]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, bool]:
        """
        Returns
        -------
        (loss, reward, hit_rate, done)
        loss: shape (1,)
            critic loss
        reward: shape (len(user_ids),)
            advantage function in actor-critic
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
        state = self.convert_state(state)
        value: torch.FloatTensor = self.critic(state, hidden_state)[1].squeeze(1)

        for _ in range(50):
            # step
            hidden_state, action_prob = self.actor(state, hidden_state)
            action_prob = F.softmax(action_prob, dim=1)
            hidden_state = hidden_state.detach()
            chosen_action, feedback_reward, in_progress = simulator.step(action_prob.detach())
            
            state = self.convert_state(chosen_action.float())
            
            # calculate advantage
            behavior_V = value.detach()
            value = self.critic(state, hidden_state)[1].squeeze(1)
            target_V = feedback_reward + self.gamma*value

            # train actor
            actor_loss = -torch.log(
                action_prob[torch.arange(len(user_ids)), chosen_action] + 1e-8
            ) * target_V.detach()
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


    def evaluate(
        self, simulator: EvaluationSimulator, user_ids: np.ndarray[np.int32]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, bool]:
        """
        Returns
        -------
        (avg_rating, hit_rate, done)
        rating: shape (len(user_ids),)
            average rating
        hit_rate:  shape (len(user_ids),)
            corresponds to hit@N
        """
        episode_avg_rating = torch.zeros(
            (len(user_ids),), dtype=torch.float32
        )
        episode_hit_rate = torch.zeros(
            (len(user_ids),), dtype=torch.float32
        )
        episode_length = torch.zeros(
            (len(user_ids),), dtype=torch.float32
        )

        state: torch.FloatTensor = torch.FloatTensor(
            simulator.reset(user_ids)
        ).to(self.device)
        hidden_state: torch.FloatTensor = torch.zeros(
            (len(user_ids), self.rnn_size), dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            state = self.convert_state(state)

            for _ in range(50):
                # step
                hidden_state, action_prob = self.actor(state, hidden_state)
                action_prob = F.softmax(action_prob, dim=1)
                chosen_action, feedback_reward, rating_reward, in_progress = simulator.step(action_prob)
            
                state = self.convert_state(chosen_action.float())

                # recording
                in_progress = in_progress.cpu()
                episode_avg_rating[in_progress] += rating_reward[in_progress]
                episode_hit_rate[in_progress] += torch.where(feedback_reward[in_progress]>0, 1, 0)
                episode_length[in_progress] += 1

                if not torch.any(in_progress):
                    break

        episode_length = torch.where(episode_length == 0, 1, episode_length)

        return (
            episode_avg_rating/episode_length,
            episode_hit_rate/episode_length,
            True
        )
    

    def save_model(self, path: str) -> bool:
        if self.use_input_embedding:
            self.item_embedding.save_model(path)
        torch.save(self.actor.state_dict(), get_path(f"{path}/actor.pth"))
        torch.save(self.critic.state_dict(), get_path(f"{path}/critic.pth"))
        return True
    

    def load_model(self, path: str) -> bool:
        if self.use_input_embedding:
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