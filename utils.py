from typing import Union
import torch
from torch import Tensor
from dataclasses import dataclass
from torchaudio.transforms import Spectrogram


def power_to_db_scale(x: Tensor) -> Tensor:
    """Converts the given input power Tensor from the
    power scale to the decibel scale

    Args:
        x (Tensor): The power scale Tensor to be converted
        to dB scale.

    Returns:
        Tensor: dB scaled tensor
    """
    return 10 * torch.log10(x)


@dataclass
class MinMax:
    min_val: Union[float, int]
    max_val: Union[float, int]


class SNR:
    def __init__(
            self,
            sample_rate: int,
            win_length: int,
            hop_length: int,
            min_snr: float,
            max_snr: float,
            ) -> None:
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.spec = Spectrogram(
            n_fft=self.win_length,
            win_length=self.win_length,
            hop_length=self.hop_length
        )

    def get_signal_power(self, x: Tensor) -> Tensor:
        """Calculates the power for the given signal x.

        Args:
            x (Tensor): Speech signal to calculate its power.

        Returns:
            Tensor: The power of the singal at each time step
        """
        result = self.spec(x)
        result = result.sum(axis=1)
        result = power_to_db_scale(result)
        return result

    def calc_snr(
            self, signal_power: Tensor, noise_power: Tensor
            ) -> Tensor:
        """Calculates the SNR based on the given signal power and the noise power.

        Args:
            signal_power (Tensor): The decibel scale Power
            of the signal of shape (1, T).
            noise_power (Tensor): The decibel scale power of
            the noise signal of shape (1, T).

        Returns:
            Tensor: The SNR at each time step T
        """
        return signal_power - noise_power

    def clip_snr(self, snr: Tensor) -> Tensor:
        return torch.clamp(snr, min=self.min_snr, max=self.max_snr)

    def __call__(self, signal: Tensor, noise: Tensor) -> Tensor:
        signal_power = self.get_signal_power(signal)
        noise_power = self.get_signal_power(noise)
        snr = self.calc_snr(signal_power, noise_power)
        return self.clip_snr(snr)
