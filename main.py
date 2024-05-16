import sys; sys.path.append(".")
from set_params import (
    set_hparams_by_argparse,
    load_model_params,
    load_dataset_params,
    adjust_parameters
)
from utils import get_path
from simulator.item_embedding import ItemEmbeddingSimulator
from simulator.feedback_based import FeedbackSimulator
from simulator.rating_based import RatingSimulator
from simulator.evaluation import EvaluationSimulator

from main_network.embedding import ItemEmbeddingModel
from main_network.feedback_based.PG import PG_Agent
from main_network.feedback_based.actor_critic import ActorCritic
from main_network.item_approximation.DDPG import DDPG


from running.item_embedding import item_embedding_training
from running.model import model_training, model_testing

import os
import json

import torch




device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

data_files = {
    'netflix_prize' : {
        'train' : [f'train_{i}' for i in range(8)],
        'test' : [f'test_{i}' for i in range(2)]
    },
    'movielens_25m' : {
        'train' : [f'train_{i}' for i in range(4)],
        'test' : [f'test_{i}' for i in range(1)]
    },
    'yahoo_r2' : {
        'train' : [f'train_{i}' for i in range(10)],
        'test' : [f'test_{i}' for i in range(10)]
    }
}




def save_hyperparameters(args: dict, model_params: dict, dataset_params: dict):
    hyperparameters = {
        'args' : args,
        'model_params' : model_params,
        'dataset_params' : dataset_params
    }

    with open(get_path(f"{args['save_folder']}/{args['target_model']}_hyperparameters.json"), 'w') as f:
        json.dump(hyperparameters, f, indent=4)




def run_item_embedding(args: dict, dataset_params: dict, model_params: dict):
    print("Item imbedding training")
    print(f"Target dataset: {args['dataset']} - {args['filename']}")
    print(f"Training under {model_params['epoch']} epochs with batch size {model_params['batch_size']}\n")

    torch.manual_seed(model_params['seed'])    # torch seed
    item_embedder = ItemEmbeddingModel(model_params, device)

    if hasattr(args, 'load_folder'):
        if item_embedder.load_model(args['load_folder']):
            print(f"Load model from {args['load_folder']}")
    else:
        print("Train a new model")

    if hasattr(args, 'save_folder'):
        if not os.path.exists(args['save_folder']):
            os.makedirs(args['save_folder'])
        print(f"Save model to {args['save_folder']}")
    else:
        print("Save path is not designated. The model will not be saved.")
    print()

    if args['filename'] in data_files[args['dataset']]:
        print(f"Given file name '{args['filename']}' consists of following files: {data_files[args['dataset']][args['filename']]}")
        epoch = model_params['epoch']
        model_params['epoch'] = 1
        data_file = data_files[args['dataset']][args['filename']]   # temp
        for e in range(epoch):
            for f in data_file:
                # if e == 0 and f not in ['train_8', 'train_9']:
                #     continue

                dataset_params['rating_file'] = f
                simulator = ItemEmbeddingSimulator(dataset_params, device)
                print(f"{f} data Info: users={len(simulator.users)}")

                item_embedding_training(item_embedder, args, model_params, simulator, verbose=True)
                
                print(f"Epoch {e+1} in progress")
            print(f"\nEpoch {e+1} done\n")
    
    else:
        simulator = ItemEmbeddingSimulator(dataset_params, device)
        print(f"Data Info: users={len(simulator.users)}")

        item_embedding_training(item_embedder, args, model_params, simulator, verbose=True)




def run_model_train(args: dict, dataset_params: dict, model_params: dict):
    print(f"Target dataset: {args['dataset']} - {args['filename']}")
    print(f"Training under {model_params['epoch']} epochs with batch size {model_params['batch_size']}\n")


    torch.manual_seed(model_params['seed'])    # torch seed

    if args['target_model'] == 'pg':
        agent = PG_Agent(model_params, device)
    elif args['target_model'] == 'actor_critic':
        agent = ActorCritic(model_params, device)
    elif args['target_model'] == 'ddpg':
        agent = DDPG(model_params, device)
    else:
        raise ValueError(f"Unsupported target model: {args['target_model']}")


    if args['load_folder'] is not None:
        if agent.load_model(args['load_folder']):
            print(f"Load model from {args['load_folder']}")
    else:
        print("Train a new model")

    if args['save_folder'] is not None:
        if not os.path.exists(args['save_folder']):
            os.makedirs(args['save_folder'])
        save_hyperparameters(args, model_params, dataset_params)
        print(f"Save model to {args['save_folder']}")
    else:
        print("Save path is not designated. The model will not be saved.")
    print()

    
    if args['filename'] in data_files[args['dataset']]:
        data_file = data_files[args['dataset']][args['filename']]
        print(f"Given file name '{args['filename']}' consists of following files: {data_file}")
        epoch = model_params['epoch']
        model_params['epoch'] = 1
        for e in range(epoch):
            for f in data_file:
                dataset_params['rating_file'] = f

                if args['reward_method'] == 'rating':
                    simulator = RatingSimulator(dataset_params, device)
                else:
                    simulator = FeedbackSimulator(dataset_params, device)
                print(f"{f} data Info: users={len(simulator.users)}, items={len(simulator.items)}")
                
                model_training(agent, args, model_params, simulator, verbose=True)
                
                print(f"Epoch {e+1} in progress")
            print(f"\nEpoch {e+1} done\n")
    
    else:
        if args['reward_method'] == 'rating':
            simulator = RatingSimulator(dataset_params, device)
        else:
            simulator = FeedbackSimulator(dataset_params, device)
        print(f"{f} data Info: users={len(simulator.users)}, items={len(simulator.items)}")

        model_training(agent, args, model_params, simulator, verbose=True)




def run_model_test(args: dict, dataset_params: dict, model_params: dict):
    print(f"Target dataset: {args['dataset']} - {args['filename']}")
    print(f"Testing under {model_params['epoch']} epochs with batch size {model_params['batch_size']}\n")


    torch.manual_seed(model_params['seed'])    # torch seed

    if args['target_model'] == 'pg':
        agent = PG_Agent(model_params, device)
    elif args['target_model'] == 'actor_critic':
        agent = ActorCritic(model_params, device)
    elif args['target_model'] == 'ddpg':
        agent = DDPG(model_params, device)
    else:
        raise ValueError(f"Unsupported target model: {args['target_model']}")


    assert args['load_folder'] is not None, "Load folder is not designated."
    if agent.load_model(args['load_folder']):
        print(f"Load model from {args['load_folder']}")
    else:
        raise FileNotFoundError(f"Model file not found in {args['load_folder']}.")

    if args['save_folder'] is not None:
        if not os.path.exists(args['save_folder']):
            os.makedirs(args['save_folder'])
        save_hyperparameters(args, model_params, dataset_params)
        print(f"Save model to {args['save_folder']}")
    else:
        print("Save path is not designated. The model will not be saved.")
    print()

    model_params['epoch'] = 1
    if args['filename'] in data_files[args['dataset']]:
        data_file = data_files[args['dataset']][args['filename']]
        print(f"Given file name '{args['filename']}' consists of following files: {data_file}")
        for f in data_file:
            dataset_params['rating_file'] = f

            simulator = EvaluationSimulator(dataset_params, device)
            print(f"{f} data Info: users={len(simulator.users)}, items={len(simulator.items)}")
            
            model_testing(agent, args, model_params, simulator, verbose=True)
    
    else:
        simulator = EvaluationSimulator(dataset_params, device)
        print(f"{f} data Info: users={len(simulator.users)}, items={len(simulator.items)}")

        model_testing(agent, args, model_params, simulator, verbose=True)




def run():
    args = set_hparams_by_argparse()
    dataset_params = load_dataset_params(args)
    model_params = load_model_params(args['target_model'])
    adjust_parameters(args, dataset_params, model_params)

    if args['target_model'] == 'embedding':
        run_item_embedding(args, dataset_params, model_params)
    
    elif args['mode'] == 'train':
        run_model_train(args, dataset_params, model_params)
        
    elif args['mode'] == 'test':
        run_model_test(args, dataset_params, model_params)




if __name__ == "__main__":
    run()