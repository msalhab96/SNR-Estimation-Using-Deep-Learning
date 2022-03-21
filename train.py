from data import AudioPipeline, NoisedAudPipeline, dataset, get_data_loader
from torch.utils.data import DataLoader
from typing import Callable, Tuple
from torch.optim import Optimizer
from torch.nn import Module
from functools import wraps
from hprams import hprams
from utils import SNR, MinMax
from model import Model
from tqdm import tqdm
import torch
import os


OPT = {
    'adam': torch.optim.Adam
}
LOSS = {
    'mae': torch.nn.L1Loss(),
    'mse': torch.nn.MSELoss()
}


def save_checkpoint(func, *args) -> Callable:
    """Save a checkpoint after each iteration
    """
    @wraps(func)
    def wrapper(obj, *args, _counter=[0], **kwargs):
        _counter[0] += 1
        result = func(obj, *args, **kwargs)
        if not os.path.exists(hprams.training.checkpoints_dir):
            os.mkdir(hprams.training.checkpoints_dir)
        model_path = os.path.join(
            hprams.training.checkpoints_dir,
            'checkpoint_' + str(_counter[0]) + '.pt'
            )
        torch.save(obj.model.state_dict(), model_path)
        print(f'checkpoint saved to {model_path}')
        return result
    return wrapper


class Trainer:
    __train_loss_key = 'train_loss'
    __test_loss_key = 'test_loss'

    def __init__(
            self,
            criterion: Module,
            optimizer: Optimizer,
            model: Module,
            device: str,
            train_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int,
            ) -> None:
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model.cuda()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.step_history = dict()
        self.history = dict()

    def fit(self):
        """The main training loop that train the model on the training
        data then test it on the test set and then log the results
        """
        for _ in range(self.epochs):
            self.train()
            self.test()
            self.print_results()

    def set_train_mode(self) -> None:
        """Set the models on the training mood
        """
        self.model = self.model.train()

    def set_test_mode(self) -> None:
        """Set the models on the testing mood
        """
        self.model = self.model.eval()

    def print_results(self):
        """Prints the results after each epoch
        """
        result = ''
        for key, value in self.history.items():
            result += f'{key}: {str(value[-1])}, '
        print(result[:-2])

    def test(self):
        """Iterate over the whole test data and test the models
        for a single epoch
        """
        total_loss = 0
        self.set_test_mode()
        for (x, y, lengths) in tqdm(self.test_loader):
            x = x.permute(0, 2, 1)
            x = x.to(self.device)
            y = y.to(self.device)
            preds = self.model(x, lengths)
            preds = preds.squeeze()
            y = y[:, :preds.shape[1]]
            loss = self.criterion(y, preds)
            total_loss += loss.item()
        total_loss /= len(self.train_loader)
        if self.__test_loss_key in self.history:
            self.history[self.__test_loss_key].append(total_loss)
        else:
            self.history[self.__test_loss_key] = [total_loss]

    @save_checkpoint
    def train(self):
        """Iterates over the whole training data and train the models
        for a single epoch
        """
        total_loss = 0
        self.set_train_mode()
        for (x, y, lengths) in tqdm(self.train_loader):
            x = x.permute(0, 2, 1)
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(x, lengths)
            preds = preds.squeeze()
            y = y[:, :preds.shape[1]]
            loss = self.criterion(y, preds)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        total_loss /= len(self.train_loader)
        if self.__train_loss_key in self.history:
            self.history[self.__train_loss_key].append(total_loss)
        else:
            self.history[self.__train_loss_key] = [total_loss]


def load_model(*args, **kwargs) -> Module:
    model = Model(**hprams.model)
    if hprams.checkpoint is not None:
        model.load_state_dict(torch.load(hprams.checkpoint))
    return model


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


def get_scalers() -> dict:
    return {
        'chunk_length': MinMax(
            hprams.data.lengths.min_val,
            hprams.data.lengths.max_val
            ),
        'signal_scaler': MinMax(
            hprams.data.signal_scaler.min_val,
            hprams.data.signal_scaler.max_val
            ),
        'noise_scaler': MinMax(
            hprams.data.noise_scaler.min_val,
            hprams.data.noise_scaler.max_val
            )
    }


def get_pipelines() -> dict:
    return {
        'aud_pipeline': AudioPipeline(
            hprams.data.sampling_rate
            ),
        'noisy_pipeline': NoisedAudPipeline(
            sample_rate=hprams.data.sampling_rate,
            n_mfcc=hprams.data.n_mfcc,
            melkwargs=get_melkwargs()
            )
    }


def get_dataset_params(data_dir: str, seed=None) -> dict:
    return dict(
        **get_pipelines(),
        **get_scalers(),
        snr_calc=SNR(**get_snr_params()),
        noise_dir=hprams.data.noise_dir,
        audio_dir=data_dir,
        seed=seed
    )


def get_train_test_loaders() -> Tuple[DataLoader, DataLoader]:
    train_loader = get_data_loader(
        batch_size=hprams.training.batch_size,
        dataset=dataset(
            **get_dataset_params(
                data_dir=hprams.data.training_dir,
                seed=hprams.data.train_seed
                )
            )
        )
    test_loader = get_data_loader(
        batch_size=hprams.training.batch_size,
        dataset=dataset(
            **get_dataset_params(
                data_dir=hprams.data.testing_dir,
                seed=hprams.data.test_seed
                )
            )
        )
    return (
        train_loader,
        test_loader
    )


def get_trainer() -> Trainer:
    device = hprams.device
    criterion = LOSS[hprams.training.loss_func]
    model = Model(**hprams.model).to(device)
    optimizer = OPT[hprams.training.optimizer](
        model.parameters(),
        lr=hprams.training.learning_rate
        )
    train_loader, test_loader = get_train_test_loaders()
    return Trainer(
        criterion=criterion,
        optimizer=optimizer,
        model=model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=hprams.training.epochs
    )


if __name__ == '__main__':
    trainer = get_trainer()
    trainer.fit()
