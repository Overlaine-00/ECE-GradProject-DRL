from utils import get_path

from argparse import ArgumentParser
from configparser import ConfigParser




model_trained_folder = get_path("./trained_models")
model_params_folder = get_path("./data/model_params")
dataset_params_folder = get_path("./data/dataset_params")


def set_hparams_by_argparse():
    parser = ArgumentParser()
    # top commands
    parser.add_argument('--load_folder', type=str, required=False, #
                        help="folder name only: excluding parent directory")
    parser.add_argument('--save_folder', type=str, required=False, #
                        help="folder name only: excluding parent directory")
    parser.add_argument('--mode', type=str, required=False, default='train', #
                        choices=['train', 'test'])

    # model related
    parser.add_argument('--target_model', type=str, required=True, #
                        choices=['embedding', 'actor_critic', 'pg', 'ddpg'])
    parser.add_argument('--reward_method', type=str, required=False, default='feedback', #
                        choices=['feedback', 'rating'])
    
    parser.add_argument('--action_type', type=str, required=False, choices=['stochastic', 'deterministic']) #
    parser.add_argument('--action_space', type=str, required=False, choices=['discrete', 'continuous']) #

    parser.add_argument('--seed', type=int, required=False, default=42)
    parser.add_argument('--epoch',type=int, required=False, default=20)

    parser.add_argument('--batch_size', type=int, required=False)

    parser.add_argument('--use_embedding', type=int, required=False, default=True)     # 0: False, 1: True
    parser.add_argument('--item_embedding_model', type=str, required=False,
                        choices=['skip_gram', 'cbow'])

    # dataset related
    parser.add_argument('--dataset', type=str, required=True)    # movielens_25m, netflix_prize, ...
    parser.add_argument('--filename', type=str, required=True)    # file name of rating data

    parser.add_argument('--episode_length', type=int, required=False)

    args = parser.parse_args()
    args = vars(args)
    
    if hasattr(args, 'use_embedding'):
        args['use_embedding'] = bool(args['use_embedding'])
    return args


def load_model_params(filename: str):
    """
    filename: exclude parent directory and '.properties'
    """
    params = dict()

    filepath = get_path(f"{model_params_folder}/{filename}.properties")
    with open(filepath, 'r') as f:
        config = ConfigParser()
        config.read_file(f)
        conf = dict(config.items("hyperparameters"))
    
    params['action_type'] = str(conf['action_type'])
    params['action_space'] = str(conf['action_space'])

    params['batch_size'] = int(conf['batch_size'])

    params['gamma'] = float(conf['gamma'])
    params['epsilon'] = float(conf['epsilon'])
    params['learning_rate'] = float(conf['learning_rate'])

    params['lr_decay_step'] = int(conf['lr_decay_step'])
    params['epsilon_decay_step'] = int(conf['epsilon_decay_step'])

    params['num_rnn_layer'] = int(conf['num_rnn_layer'])
    params['rnn_state_dim'] = int(conf['rnn_state_dim'])

    params['top_n'] = int(conf['top_n'])

    return params


def load_dataset_params(args: dict):
    """
    filename: exclude parent directory and '.properties'
    """
    params = dict()

    # load data (user rating)
    filepath = get_path(f"{model_params_folder}/{args['dataset']}.properties")

    params['rating'] = args['dataset']
    params['rating_file'] = args['filename']

    with open(filepath, 'r') as f:
        config = ConfigParser()
        config.read_file(f)
        conf = dict(config.items("hyperparameters"))

    params['num_users'] = int(conf['num_users'])
    params['num_items'] = int(conf['num_items'])
    params['episode_length'] = int(conf['episode_length'])

    with open(filepath, 'r') as f:
        config = ConfigParser()
        config.read_file(f)
        conf = dict(config.items("embedding"))

    params['item_embedding_model'] = conf['item_embedding_model']
    params['context_size'] = int(conf['context_size'])
    params['state_embedding_dim'] = int(conf['state_embedding_dim'])

    return params


def adjust_parameters(args: dict, dataset_params: dict, model_params: dict):
    """
    1. Hparams `args` has the highest priority.  
        Set the corresponding hparams in `model_params` and `dataset_params` to the value in `args`.
    2. Some hparams in `dataset_params` are required in `model_params`.  
        Set the corresponding hparams in `model_params` to the value in `dataset_params`.
        ex) hparams related to state, action, reward, etc.
    """
    ## change args
    if args['load_folder'] is not None:
        args['load_folder'] = get_path(f"{model_trained_folder}/{args['load_folder']}")
    if args['save_folder'] is not None:
        args['save_folder'] = get_path(f"{model_trained_folder}/{args['save_folder']}")
    if args['mode'] == 'test':
        args['epoch'] = 1

    ## args -> others
    model_params['seed'] = args['seed']
    model_params['epoch'] = args['epoch']
    model_params['use_embedding'] = args['use_embedding']

    # model related
    if hasattr(args, 'batch_size'):
        model_params["batch_size"] = args['batch_size']
    
    if hasattr(args, 'action_type'):
        model_params['action_type'] = args['action_type']
    if hasattr(args, 'action_space'):
        model_params['action_space'] = args['action_space']

    # dataset related
    if hasattr(args, 'item_embedding_model'):
        dataset_params['item_embedding_model'] = args['item_embedding_model']

    ## dataset_params -> model_params
    model_params['num_items'] = dataset_params['num_items']
    model_params['state_maxlength'] = dataset_params['episode_length']

    model_params['context_size'] = dataset_params['context_size']
    model_params['state_embedding_dim'] = dataset_params['state_embedding_dim']
    model_params['item_embedding_model'] = dataset_params['item_embedding_model']
    

    ## model_params -> dataset_params
    dataset_params['action_type'] = model_params['action_type']
    dataset_params['action_space'] = model_params['action_space']

    dataset_params['epsilon'] = model_params['epsilon']
    dataset_params['epsilon_decay_step'] = model_params['epsilon_decay_step']
    dataset_params['top_n'] = model_params['top_n']