import torch
from torch import Tensor
from torch.nn import Module
from data import (
    AudioPipeline, 
    NoisedAudPipeline
    )
from utils import SNR
from train import get_snr_params


class Predict:
    def __init__(
            self,
            noised_pipeline: NoisedAudPipeline,
            model: Module,
            max_length: int,
            device: str
            ) -> None:
        self.noised_pipeline = noised_pipeline
        self.max_length = max_length
        self.model = model.to(device)
        self.device = device

    def predict(self, signal: Tensor):
        signal = self.noised_pipeline.run(signal)
        signal = signal.permute(0, 2, 1)
        return self.model(signal, [signal.shape[1]])
