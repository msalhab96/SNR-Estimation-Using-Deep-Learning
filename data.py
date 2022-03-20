import os
import random
from typing import Callable, Tuple
import torch
from torch import Tensor
import torchaudio
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from utils import MinMax


class IPipeline(ABC):
    @abstractmethod
    def run():
        """Used to run all the callables functions sequantially
        """
        pass


class AudioPipeline(IPipeline):
    def __init__(self, target_sr: int) -> None:
        super().__init__()
        self.target_sr = target_sr

    def get_resampler(self, sr: int):
        return torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=self.target_sr
        )

    def run(self, filepath: str) -> Tensor:
        x, sr = torchaudio.load(filepath)
        return self.get_resampler(sr)(x)


class NoisedAudPipeline(IPipeline):
    def __init__(self) -> None:
        super().__init__()
        # TODO


class dataset(Dataset):
    def __init__(
            self,
            snr_calc: Callable,
            aud_pipeline: IPipeline,
            noisy_pipeline: IPipeline,
            noise_dir: str,
            audio_dir: str,
            chunk_length: MinMax,
            signal_scaler: MinMax,
            noise_scaler: MinMax,
            seed=None
            ) -> None:
        super().__init__()
        self.snr_calc = snr_calc
        self.noise_files = [
            os.path.join(noise_dir, file) for file in os.listdir(noise_dir)
            ]
        self.audio_files = [
            os.path.join(audio_dir, file) for file in os.listdir(audio_dir)
            ]
        self.chunk_length = chunk_length
        self.signal_scaler = signal_scaler
        self.noise_scaler = noise_scaler
        self.seed = seed
        self.aud_pipeline = aud_pipeline
        self.noisy_pipeline = noisy_pipeline

    def get_noise(self) -> Tensor:
        noise_file = random.choice(self.noise_files)
        return self.aud_pipeline.run(noise_file)

    def get_signal(self, index: int) -> Tensor:
        signal_file = self.audio_files[index]
        return self.aud_pipeline.run(signal_file)

    def get_scalers(self) -> Tuple[float, float]:
        return (
            random.uniform(
                self.signal_scaler.min_val, self.signal_scaler.max_val
                ),
            random.uniform(
                self.noise_scaler.min_val, self.noise_scaler.max_val
            )
        )

    def pick_signal_chunk(self, signal: Tensor) -> Tensor:
        if self.seed is not None:
            random.seed(self.seed)
        length = signal.shape[1]
        if length <= self.chunk_length.min_val:
            return signal
        max_val = length - self.chunk_length.min_val
        first_index = random.randrange(0, max_val)
        offset = random.randrange(
            self.chunk_length.min_val, self.chunk_length.max_val
            )
        return signal[:, first_index: min(length, first_index + offset)]

    def pick_noisy_chunk(
            self, signal: Tensor, noise: Tensor
            ) -> Tuple[Tensor, Tensor]:
        noise_signal = torch.zeros_like(signal)
        noise_length = noise.shape[1]
        signal_length = signal.shape[1]
        if noise_length == signal_length:
            return (signal + noise), noise
        elif noise_length < signal_length:
            diff = signal_length - noise_length
            start_idx = random.randrange(0, diff)
            noise_signal[:, start_idx:] += noise
            return (signal + noise_signal), noise_signal
        else:
            diff = noise_length - signal_length
            start_idx = random.randrange(0, diff)
            noise_signal = noise[:, start_idx: start_idx + signal_length]
            return (signal + noise_signal), noise_signal

    def __getitem__(self, index):
        noise = self.get_noise()
        signal = self.get_signal(index)
        signal_scaler, noise_scaler = self.get_scalers()
        signal = self.pick_signal_chunk(signal)
        noise *= noise_scaler
        signal *= signal_scaler
        noisy_signal, noise = self.pick_noisy_chunk(signal, noise)
        snr = self.snr_calc(signal, noise)
        # TODO: Padd the results
        return self.noisy_pipeline(noisy_signal), snr

    def __len__(self):
        return len(self.audio_files)