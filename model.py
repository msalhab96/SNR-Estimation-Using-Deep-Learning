from typing import List, Union
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):
    def __init__(
            self, 
            inp_size: int, 
            num_layers: int, 
            hidden_size: int,
            is_causal: bool
            ) -> None:
        """
        Args:
            inp_size (int): The shape of the input
            num_layers (int): The number of LSTM/biLSTM layers
            hidden_size (int): The hidden size of the LSTMs/biLSTMs 
            is_causal (bool): If True at time t the model will only look at 
            the frames from 0 till t-1 (LSTM), otherwise will be looking at 
            the previous and the future frames (biLSTM)
        """
        super().__init__()
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=inp_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional= not is_causal
        )
        self.output_layer = nn.Linear(
            in_features=hidden_size if is_causal else 2 * hidden_size,
            out_features=1
        )
    
    def forward(
            self, 
            x: Tensor, 
            lengths: Union[List[int], Tensor]
            ) -> Tensor:
        """Performs forward pass for the input x.

        Args:
            x (Tensor): The input to the model of shape (B, M, D)
            lengths (Union[List[int], Tensor]): The lengths of 
            the input without padding.

        Returns:
            Tensor: The estimated SNR
        """
        packed_seq = pack_padded_sequence(
            input=x,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False
        )
        output, (hn, cn) = self.lstm(packed_seq)
        seq_unpacked, _ = pad_packed_sequence(
            sequence=output,
            batch_first=True
        )
        return self.output_layer(seq_unpacked)
