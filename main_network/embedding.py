from utils import get_path
from simulator.item_embedding import ItemEmbeddingSimulator

from typing import Union

import numpy as np

import torch
import torch.nn as nn




class SkipGramModel(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int, context_size: int):
        super(SkipGramModel, self).__init__()
        self.context_size = context_size
        self.num_items = num_items

        self.embedding = nn.Embedding(num_items, embedding_dim)
        self.output = nn.Sequential(
            nn.Linear(embedding_dim, num_items),
            nn.Softmax(dim=1)
        )


    def forward(
        self, context_items: torch.LongTensor
    ) -> torch.FloatTensor:
        '''
        Parameters
        ----------
        context_items: items around the main item, of shape (batch_size,)

        Returns
        -------
        shape (batch_size, num_items), at `self.device`
        score between 0~1, and the higher the better.
        '''
        context_items = self.embedding(context_items)
        score_values = self.output(context_items)
        return score_values
    

    def get_embedding(self, items: torch.FloatTensor) -> torch.FloatTensor:
        '''
        Variable `item` corresponds to the `target item` in `forward()`.
        '''
        return self.embedding(items)
    

    def get_decoding_score(self, embedded_items: torch.FloatTensor) -> torch.FloatTensor:
        """
        Returns
        -------
        shape (batch_size,), at `self.device`
        """
        distances = torch.cdist(embedded_items, self.embedding.weight, p=2)
        # distances = torch.sum(
        #     (embedded_items.unsqueeze(1) - self.embedding.weight.unsqueeze(0))**2,
        #     dim=-1
        # )
        return -distances
        



class CBOWModel(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int):
        super(CBOWModel, self).__init__()
        self.embedding = nn.Embedding(num_items, embedding_dim)
        self.output = nn.Sequential(
            nn.Linear(embedding_dim, num_items),
            nn.Softmax(dim=1)
        )


    def forward(
        self, context_items: torch.FloatTensor
    ) -> torch.FloatTensor:
        '''
        Parameters
        ----------
        context_items: items around the main item, of shape (batch_size, context_size)

        Returns
        -------
        shape (batch_size, num_items), at `self.device`
        score between 0~1, and the higher the better.
        '''
        context_items = self.embedding(context_items)
        context_items = torch.mean(context_items, dim=1)
        target_item = self.output(context_items)
        return target_item


    def get_embeddings(self, items: torch.FloatTensor) -> torch.FloatTensor:
        return self.embedding(items)
    

    def get_decoding_score(self, emdbedded_item: torch.FloatTensor) -> torch.FloatTensor:
        """
        This does not contain `torch.mean()` in `forward()`.

        Returns
        -------
        shape (batch_size,) at `self.device`
        """
        return self.output(emdbedded_item)




class ItemEmbeddingModel:
    def __init__(self, model_params: dict, device: torch.device):
        super(ItemEmbeddingModel, self).__init__()
        self.device = device
        self.num_items = model_params['num_items']
        self.used_model = model_params['item_embedding_model']
        self.context_size = model_params['context_size']
        self.embedding_dim = model_params['state_embedding_dim']

        self.model: Union[SkipGramModel, CBOWModel]
        if self.used_model == 'skip_gram':
            self.model = SkipGramModel(self.num_items, self.embedding_dim, self.context_size).to(self.device)
            # no loss function, because the purpose is to maximize the score
        elif self.used_model == 'cbow':
            self.model = CBOWModel(self.num_items, self.embedding_dim, self.context_size).to(self.device)
            self.loss_function = nn.CrossEntropyLoss()
        else:
            raise ValueError(f'Invalid model name: {self.used_model}')
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)


    def train(
            self, simulator: ItemEmbeddingSimulator, user_ids: np.ndarray[np.int32]
        ) -> torch.FloatTensor:
        """
        Consider past contexts ONLY.

        skip-gram:
        `target_items`: shape (batch_size, context_size)
        `context_items`: shape (batch_size,)
        cbow:
        `target_items`: shape (batch_size,)
        `context_items`: shape (batch_size, context_size)

        Returns
        -------
        loss: torch.FloatTensor at CPU
        """
        target_items: torch.LongTensor
        context_items: torch.LongTensor
        target_items, context_items = simulator.get_data(self.used_model, user_ids, self.context_size)

        model_output: torch.FloatTensor = self.model(context_items)

        if self.used_model == 'skip_gram':
            # model_output is score of each items
            score = model_output[np.arange(model_output.size(0)), target_items.T].T.sum()
            loss = -score
        elif self.used_model == 'cbow':
            # model_output is probability of target items
            loss = self.loss_function(model_output, target_items)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu()


    def get_item_embedding(self, item: torch.FloatTensor) -> torch.FloatTensor:
        '''
        This function NEVER requires gradient.
        '''
        with torch.no_grad():
            return self.model.get_embedding(item)
    

    def get_original_item_scores(self, embedded_item: torch.FloatTensor) -> torch.FloatTensor:
        '''
        This function NEVER requires gradient.
        '''
        with torch.no_grad():
            return self.model.get_decoding_score(embedded_item)
    

    def save_model(self, path: str) -> bool:
        torch.save(self.model.state_dict(), get_path(f"{path}/item_embedding_{self.used_model}.pth"))
        return True


    def load_model(self, path: str) -> bool:
        try:
            self.model.load_state_dict(torch.load(get_path(f"{path}/item_embedding_{self.used_model}.pth")))
            return True
        except FileNotFoundError:
            print(f"Embedding model file not found in {path}. Train a new model.")
            return False