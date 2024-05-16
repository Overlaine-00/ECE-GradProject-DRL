from main_network.feedback_based.PG import PG_Agent
from main_network.feedback_based.actor_critic import ActorCritic
from main_network.item_approximation.DDPG import DDPG
from simulator.feedback_based import FeedbackSimulator
from simulator.rating_based import RatingSimulator
from simulator.evaluation import EvaluationSimulator
from utils import get_path

from tqdm import tqdm
from typing import Union

import numpy as np




def model_training(
    agent: Union[PG_Agent, ActorCritic, DDPG],
    args: dict,
    model_params: dict,
    simulator: Union[FeedbackSimulator, RatingSimulator],
    verbose: bool=False
):
    assert args['mode'] == 'train', \
        f"Model training requires mode train, whereas {args['mode']} is entered."
    
    batch_size = model_params['batch_size']
    epoch = model_params['epoch']
    intermediate_save = max(1, epoch // 20)

    users = simulator.users.astype(np.int64)    # index -> int64
    iteration = len(users)//batch_size

    cumulative_average_rewards = np.zeros(
        (intermediate_save*iteration, batch_size), dtype=np.float32
    )
    cumulative_hit_rates = np.zeros(
        (intermediate_save*iteration, batch_size), dtype=np.float32
    )
    losses = np.zeros(
        (intermediate_save*iteration,), dtype=np.float32
    )
    early_stop = 0

    for e in range(epoch):
        np.random.shuffle(users)

        iteration_range = range(iteration)
        if verbose: iteration_range = tqdm(iteration_range)

        for i in iteration_range:
            user_ids = users[i*batch_size:(i+1)*batch_size]

            loss, reward, hit, done = agent.train(simulator, user_ids)
            
            idx = (e%intermediate_save)*iteration+i
            cumulative_average_rewards[idx, :] = reward.numpy()
            cumulative_hit_rates[idx, :] = hit.numpy()
            losses[idx] = loss.numpy()
            early_stop += done
            

        if ((e+1)%intermediate_save == 0) or (e+1 == epoch):
            print(f'Epoch {e+1}/{epoch}, loss: {losses[idx]:.4f}, cumulative reward: {np.mean(cumulative_hit_rates[idx, :]):.4f}, early_stop: {early_stop}/{iteration}')
            agent.save_model(args['save_folder'])

            try:
                cumulative_average_rewards = np.concatenate([
                    np.load(get_path(f"{args['save_folder']}/cumulative_rewards.npy")),
                    cumulative_average_rewards
                ], axis=0)
            except FileNotFoundError:
                pass
            np.save(get_path(f"{args['save_folder']}/cumulative_rewards.npy"), cumulative_average_rewards)

            try:
                cumulative_hit_rates = np.concatenate([
                    np.load(get_path(f"{args['save_folder']}/cumulative_hit_rates.npy")),
                    cumulative_hit_rates
                ], axis=0)
            except FileNotFoundError:
                pass
            np.save(get_path(f"{args['save_folder']}/cumulative_hit_rates.npy"), cumulative_hit_rates)

            try:
                losses = np.concatenate([
                    np.load(get_path(f"{args['save_folder']}/losses.npy")),
                    losses
                ], axis=0)
            except FileNotFoundError:
                pass
            np.save(get_path(f"{args['save_folder']}/losses.npy"), losses)
            
            cumulative_average_rewards[:,:] = 0
            cumulative_hit_rates[:,:] = 0
            losses[:] = 0




def model_testing(
    agent: Union[PG_Agent, ActorCritic, DDPG],
    args: dict,
    model_params: dict,
    simulator: EvaluationSimulator,
    verbose: bool=False
):
    assert args['mode'] == 'test', \
        f"Model testing requires mode test, whereas {args['mode']} is entered."
    
    batch_size = model_params['batch_size']
    epoch = model_params['epoch']
    intermediate_save = max(1, epoch // 20)

    users = simulator.users.astype(np.int64)    # index -> int64
    iteration = len(users)//batch_size
    
    cumulative_average_ratings = np.zeros(
        (intermediate_save*iteration, batch_size), dtype=np.float32
    )
    cumulative_hit_rates = np.zeros(
        (intermediate_save*iteration, batch_size), dtype=np.float32
    )

    for e in range(epoch):
        np.random.shuffle(users)

        iteration_range = range(iteration)
        if verbose: iteration_range = tqdm(iteration_range)

        for i in iteration_range:
            user_ids = users[i*batch_size:(i+1)*batch_size]

            rating, hit, _ = agent.evaluate(simulator, user_ids)
            
            idx = (e%intermediate_save)*iteration+i
            cumulative_average_ratings[idx, :] = rating.numpy()
            cumulative_hit_rates[idx, :] = hit.numpy()
            

        if ((e+1)%intermediate_save == 0) or (e+1 == epoch):
            print(f'Epoch {e+1}/{epoch}, reward: {np.mean(cumulative_hit_rates[idx, :]):.4f}, hit rate: {np.mean(cumulative_hit_rates[idx, :]):.4f}')
            agent.save_model(args['save_folder'])

            try:
                cumulative_average_ratings = np.concatenate([
                    np.load(get_path(f"{args['save_folder']}/performance_ratings.npy")),
                    cumulative_average_ratings
                ], axis=0)
            except FileNotFoundError:
                pass
            np.save(get_path(f"{args['save_folder']}/performance_ratings.npy"), cumulative_average_ratings)

            try:
                cumulative_hit_rates = np.concatenate([
                    np.load(get_path(f"{args['save_folder']}/performance_hit_rates.npy")),
                    cumulative_hit_rates
                ], axis=0)
            except FileNotFoundError:
                pass
            np.save(get_path(f"{args['save_folder']}/performance_hit_rates.npy"), cumulative_hit_rates)
            
            cumulative_average_ratings[:,:] = 0
            cumulative_hit_rates[:,:] = 0
