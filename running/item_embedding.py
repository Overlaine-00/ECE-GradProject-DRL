from main_network.embedding import ItemEmbeddingModel
from simulator.item_embedding import ItemEmbeddingSimulator
from utils import get_path

from tqdm import tqdm

import numpy as np




def item_embedding_training(
    model: ItemEmbeddingModel,
    args: dict,
    model_params: dict,
    simulator: ItemEmbeddingSimulator,
    verbose: bool=False
):
    batch_size = model_params['batch_size']
    epoch = model_params['epoch']
    intermediate_save = max(1, epoch // 20)

    users = simulator.users.astype(np.int64)    # index -> int64
    iteration = len(users)//batch_size

    losses = np.zeros(
        (intermediate_save*iteration,), dtype=np.float32
    )

    for e in range(epoch):
        np.random.shuffle(users)

        iteration_range = range(iteration)
        if verbose: iteration_range = tqdm(iteration_range)

        for i in iteration_range:
            user_ids = users[i*batch_size:(i+1)*batch_size]

            loss = model.train(simulator, user_ids)
            
            idx = (e%intermediate_save)*iteration+i
            losses[idx] = loss.numpy()
            

        if ((e+1)%intermediate_save == 0) or (e+1 == epoch):
            print(f'Epoch {e+1}/{epoch}, loss: {loss:.4f}')
            model.save_model(args['save_folder'])

            try:
                losses = np.concatenate([
                    np.load(get_path(f"{args['save_folder']}/embedding_{model.used_model}_losses.npy")),
                    losses
                ], axis=0)
            except FileNotFoundError:
                pass
            np.save(get_path(f"{args['save_folder']}/embedding_{model.used_model}_losses.npy"), losses)
            
            losses[:] = 0