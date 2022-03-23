from hydra import compose, initialize
from sys import argv
initialize(config_path="config")
hprams = compose(config_name="configs", overrides=argv[1:])


def get_melkwargs() -> dict:
    return {
        'n_fft': hprams.data.n_fft,
        'win_length': hprams.data.win_length,
        'hop_length': hprams.data.hop_length
    }


def get_snr_params() -> dict:
    return {
        'sample_rate': hprams.data.sampling_rate,
        'win_length': hprams.data.win_length,
        'hop_length': hprams.data.hop_length,
        'min_snr': hprams.data.min_snr,
        'max_snr': hprams.data.max_snr
    }
