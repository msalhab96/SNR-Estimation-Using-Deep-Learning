import torch
from torch import Tensor
from torch.nn import Module
from data import (
    NoisedAudPipeline,
    AudioPipeline
    )
from utils import SNR, load_model, join
from train import get_pipelines
from  hprams import hprams, get_snr_params

class Predict:
    def __init__(
            self,
            noised_pipeline: NoisedAudPipeline,
            model: Module,
            device: str
            ) -> None:
        self.noised_pipeline = noised_pipeline
        self.model = model.to(device)
        self.device = device

    def predict(self, signal: Tensor):
        signal = self.noised_pipeline.run(signal)
        signal = signal.permute(0, 2, 1)
        return self.model(signal, torch.tensor([signal.shape[1]]))


def get_predictor() -> Predict:
    model = load_model(hprams.model, hprams.checkpoint, device=hprams.device)
    noised_pipeline = get_pipelines()['noisy_pipeline']
    return Predict(
        noised_pipeline,
        model,
        hprams.device
    )


if __name__ == '__main__':
    predictor = get_predictor()
    noise_path = 'samples/noise/constructionSite_bg21.wav'
    audio_path = 'samples/audios/19-198-0005.flac'
    aud_pipeline = AudioPipeline(hprams.data.sampling_rate)
    signal = aud_pipeline.run(audio_path)
    noise = aud_pipeline.run(noise_path)
    new_aud, noise = join(signal, noise)
    print(predictor.predict(new_aud))