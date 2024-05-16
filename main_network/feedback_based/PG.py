from main_network.embedding import ItemEmbeddingModel
from simulator.feedback_based import FeedbackSimulator
from simulator.evaluation import EvaluationSimulator
from main_network.NN.GRU import GRURecurrent
from utils import get_path

import numpy as np
import torch
import torch.nn.functional as F




class PG_Agent:
    def __init__(
        self,
        model_params: dict,
        device: torch.device,
    ):
        assert model_params['action_type'] == 'stochastic', \
            f"Only stochastic policy is supported in Policy Gradient, whereas {model_params['action_type']} is entered."
        assert model_params['action_space'] == 'discrete', \
            f"Only discrete action space is supported in Policy Gradient, whereas {model_params['action_space']} is entered."
        self.device = device
        self.use_input_embedding: bool = model_params['use_embedding']

        self.gamma = model_params['gamma']
        self.rnn_size = model_params['rnn_state_dim']

        self.state_dimension: int
        if self.use_input_embedding:
            self.item_embedding = ItemEmbeddingModel(
                model_params, device
            )
            self.state_dimension = model_params['state_embedding_dim']
        else:
            self.state_dimension = 1

        self.model = GRURecurrent(
            self.state_dimension, model_params['num_items'], self.rnn_size, model_params['num_rnn_layer']
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=model_params['learning_rate'])
    

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
        loss: equal to -rewards
        rewards: corresponds to V_t, shape (len(user_ids),)
            -0.2: negaative, 1: positive
        hit_rate:  shape (len(user_ids),)
            corresponds to hit@N
        done: bool
            True if the episode is done (needed to be reset), False otherwise
        """
        episode_reward = torch.zeros(
            (len(user_ids),), dtype=torch.float32
        ).to(self.device)
        episode_hit_rate = torch.zeros(
            (len(user_ids)), dtype=torch.float32
        )
        episode_length = torch.zeros(
            (len(user_ids),), dtype=torch.float32
        )
        early_stop = False

        state = torch.FloatTensor(simulator.reset(user_ids)).to(self.device)
        hidden_state = torch.zeros(
            (len(simulator.user_ids), self.rnn_size),
            dtype=torch.float32
        ).to(self.device)
        feedback_reward = torch.ones(
            (len(simulator.user_ids),),
            dtype=torch.float32
        ).to(self.device)

        state = self.convert_state(state)

        for _ in range(50):
            # step
            hidden_state, action_prob = self.model(state, hidden_state)
            action_prob = F.softmax(action_prob, dim=1)
            chosen_action, feedback_reward, in_progress = simulator.step(action_prob)

            state = self.convert_state(chosen_action.float())

            # calculate reward
            episode_reward[in_progress] *= self.gamma
            episode_reward[in_progress] += (
                torch.log(action_prob[torch.arange(len(user_ids)), chosen_action] + 1e-8) \
                * feedback_reward
            )[in_progress]

            # record hit rate
            in_progress = in_progress.cpu()
            episode_hit_rate[in_progress] += torch.where(feedback_reward.cpu()[in_progress]>0, 1, 0)
            episode_length[in_progress] += 1

            if not torch.any(in_progress):
                early_stop = True
                break
        
        loss = torch.mean(-episode_reward)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        episode_reward = episode_reward.detach().cpu()
        episode_length = torch.where(episode_length == 0, 1, episode_length)
        return -episode_reward.mean(), episode_reward, episode_hit_rate/episode_length, early_stop
    

    def evaluate(
            self, simulator: EvaluationSimulator, user_ids: np.ndarray[np.int32]
        ) -> tuple[torch.FloatTensor, torch.FloatTensor, bool]:
        """
        Returns
        -------
        (avg_rating, hit_rate, done)
        rating: shape (len(user_ids),)
            average rating \\
            at cpu
        hit_rate:  shape (len(user_ids),)
            corresponds to hit@N \\
            at cpu
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

        state = torch.FloatTensor(simulator.reset(user_ids)).to(self.device)
        hidden_state = torch.zeros(
            (len(simulator.user_ids), self.rnn_size),
            dtype=torch.float32
        ).to(self.device)


        with torch.no_grad():
            state = self.convert_state(state)

            for _ in range(50):
                # step
                hidden_state, action_prob = self.model(state, hidden_state)
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
        """
        path: parent directory (exclude '/' and file name)
        """
        if self.use_input_embedding:
            self.item_embedding.save_model(path)
        torch.save(self.model.state_dict(), get_path(f"{path}/main_network.pth"))
        return True
    

    def load_model(self, path: str) -> bool:
        if self.use_input_embedding:
            if not self.item_embedding.load_model(path):
                print("The program will be terminated.")
                exit()

        try:
            self.model.load_state_dict(torch.load(get_path(f"{path}/main_network.pth")))
            return True
        except FileNotFoundError:
            print(f"Model file not found in {path}. Train a new model.")
            return False
