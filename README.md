# SNR-Estimation-Using-Deep-Learning

PyTorch implementation of [Frame-level Signal-to-Noise Ratio Estimation using Deep Learning](https://www.isca-speech.org/archive_v0/Interspeech_2020/pdfs/2475.pdf).


This implementation includes distributed training and trained on [LibriSpeech -train-clean-100.tar.gz- dataset](https://www.openslr.org/12) and the noise collected from different sources.

# Setup
1. Download and extract [LibriSpeech](https://www.openslr.org/12)
2. Clone this repo: `git clone https://github.com/msalhab96/SNR-Estimation-Using-Deep-Learning`
3. CD into this repo: `cd SNR-Estimation-Using-Deep-Learning`
4. Install the requirements: `pip install -r requiremnts.txt`

# Training
To train the model follow the steps below:
1. preprocess all the audio files and make sure all of them are single channled audios
2. change the configuration in the config/configs.yaml file
3. run `python train.py` to train from scratch or `python train.py checkpoint=path/to/checkpoint` to train the model from a checkpoint
