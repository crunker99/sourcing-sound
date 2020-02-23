import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.io import wavfile
from python_speech_features import mfcc
import librosa
from librosa.feature import melspectrogram
import pickle
from tensorflow.keras.utils import to_categorical
from cfg import Config

# config = Config()

def check_data(config):
    if os.path.isfile(config.p_path):
        print('Loading existing data for {} model'.format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp 
    else:
        return None

def build_rand_feat(config, df, split):
    tmp = check_data(config)
    if not tmp:
        tmp = Config()
        tmp.data = defaultdict(list)
    if split == 'train' and not tmp.data[0] is None:
            return tmp.data[0], tmp.data[1]
    elif split == 'val' and not tmp.data[2] is None:
            return tmp.data[2], tmp.data[3]
    # config.data = [None, None, None, None]
    n_samples = 1 * int(df['length'].sum() / 0.1)
    classes = list(np.unique(df.labels))
    class_dist = df.groupby(['labels'])['length'].mean()
    prob_dist = class_dist / class_dist.sum()
    _min, _max = float('inf'), -float('inf')

    X = []
    y = []
    print('Building features for '+split)
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(classes, p=prob_dist)
        file = np.random.choice(df[df.labels == rand_class].index)
        rate, wav = wavfile.read('clean/'+file)
        rand_index = np.random.randint(0, wav.shape[0] - config.step)
        sample = wav[rand_index:rand_index + config.step]

        if config.feature_type == 'mfccs':
            X_sample = mfcc(sample, rate, numcep=config.nfeat,
                            nfilt=config.nfilt, nfft = config.nfft)
        elif config.feature_type == 'mels':
            X_sample = melspectrogram(sample, rate, n_mels=config.n_mels,
                                        n_fft=config.nfft)
            X_sample = librosa.power_to_db(X_sample)
        elif config.feature_type == 'raw':
            X_sample = sample

        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample if config.mode== 'conv' else X_sample.T)
        y.append(classes.index(rand_class)) # encoding integer values for classes
    config.min = _min
    config.max = _max
    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y)

    if split == 'train':
        config.data = [None, None]
        config.data[0], config.data[1] = (X, y)
    elif split == 'val':
        config.data.append(X)
        config.data.append(y)

    # with open(config.p_path, 'wb') as handle:
    #     pickle.dump(config, handle, protocol=2)
    return X, y

if __name__ == '__main__':
    pass