# SNR-Estimation-Using-Deep-Learning

PyTorch implementation of [Frame-level Signal-to-Noise Ratio Estimation using Deep Learning](https://www.isca-speech.org/archive_v0/Interspeech_2020/pdfs/2475.pdf).


This implementation includes distributed training and trained on [LibriSpeech -train-clean-100.tar.gz- dataset](https://www.openslr.org/12) and the noise collected from different sources.

The below image is taken from the training on LibriSpeech

![123](https://user-images.githubusercontent.com/61272193/159757830-be436113-5279-488a-8bf2-a614ce705936.jpg)

The below images are samples shows the results 

![image](https://user-images.githubusercontent.com/61272193/159758156-a46e0064-7bc9-4c3f-9d4f-445822368b00.png)

![image](https://user-images.githubusercontent.com/61272193/159758325-4ad53a81-71dd-4dde-90d2-c905143dc0a7.png)



# Setup
1. Download and extract [LibriSpeech](https://www.openslr.org/12)
2. Clone this repo: `git clone https://github.com/msalhab96/SNR-Estimation-Using-Deep-Learning`
3. CD into this repo: `cd SNR-Estimation-Using-Deep-Learning`
4. Install the requirements: `pip install -r requiremnts.txt`

# Training
To train the model follow the steps below:
1. Preprocess all the audio files and make sure all of them are single channeled audios
2. Change the configuration in the config/configs.yaml file
3. Run `python train.py` to train from scratch or `python train.py checkpoint=path/to/checkpoint` to train the model from a checkpoint
4. Run `tensorboard --logdir=logdir/` to monitor the training (optional)

## Pre-trained model
You can download the pretrained model from [here](https://drive.google.com/file/d/1uoiHiDiK3RY8eD4Ow0C4nLPrkkU9OZfr/view?usp=sharing)
