from data.embedding_generator import load_embedded_data

import numpy as np

import torch
from torch import LongTensor




class ItemEmbeddingSimulator:
    def __init__(self, dataset_params: dict, device: torch.device):
        self.device = device
        self.context_size = dataset_params['context_size']
        self.used_model = dataset_params['item_embedding_model']

        # load ratings file
        self.dataset_name = dataset_params['rating']
        self.file_name = dataset_params['rating_file']
        self.target, self.context = load_embedded_data(
            dataset_params['rating'], dataset_params['rating_file'], self.used_model, self.context_size
        )
        print("Rating data is loaded.")

        # parameters
        self.users = np.arange(self.target.shape[0])

        print("Simulator is ready.")
    

    def get_data(
        self, data_type: str, user_ids: np.ndarray[np.int32], context_size: int
    ) -> tuple[LongTensor, LongTensor]:
        """
        Consider past contexts ONLY.

        `padded_items`: shape (batch_size, max_length_of_items)
        `data_type == cbow`: 
            `target_items`: shape (batch_size, max_length_of_items - context_size, context_size) -> flatten (*,context_size)
            `context_items`: shape (batch_size, max_length_of_items - context_size) -> flatten (*,)
        `data_type == skip_gram`:
            `target_items`: shape (batch_size, max_length_of_items - context_size) -> flatten (*,)
            `context_items`: shape (batch_size, max_length_of_items - context_size, context_size) -> flatten (*,context_size)

        Parameters
        ----------
        data_type: 'cbow' or 'skip_gram'

        Returns
        -------
        target_items, context_items
        loaded at `device`
        """
        assert context_size == self.context_size, \
            f"Context size does not match: given {context_size}, expected {self.context_size}"
        assert data_type == self.used_model, \
            f"Data type does not match: given {data_type}, expected {self.used_model}"

        target_items = self.target[user_ids]
        context_items = self.context[user_ids]

        return (
            LongTensor(target_items).to(device=self.device),
            LongTensor(context_items).to(device=self.device)
        )