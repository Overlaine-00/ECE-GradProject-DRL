import torch
from torch import nn

class GRURecurrent(nn.Module):
    """
    This does not contain softmax at tail
    """
    def __init__(
            self,
            state_size: int,
            action_size: int,
            rnn_size: int,
            num_layer: int
        ):
        super(GRURecurrent, self).__init__()

        self.gru = nn.ModuleList(
            [nn.GRUCell(state_size, rnn_size)] + \
            [nn.GRUCell(rnn_size, rnn_size) for _ in range(num_layer-1)]
        )
        self.tail = nn.Linear(rnn_size, action_size)
    

    def forward(
        self,
        pres_state: torch.FloatTensor,
        prev_hidden_state: torch.FloatTensor
    ):
        '''
        Note: GRUCell output == hidden_state when the length of sequence is 1

        Parameters
        ----------
        pres_state: torch.FloatTensor, shape (batch_size, state_size)  
            (here we use direct previous state only)
        prev_hidden_state: torch.FloatTensor, shape (batch_size, rnn_size)

        Returns
        -------
        hidden_state, output
        shape (batch_size, rnn_size), (batch_size, state_size)
        '''

        hidden_state = prev_hidden_state
        for gru in self.gru:
            hidden_state = gru(pres_state, hidden_state)
        return hidden_state, self.tail(hidden_state)