import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa
import librosa.display

"""
Create explanatory visuals of transformation of audio signals into spectrograms.
Explore class distributions.
Pre-process training data, downsample test data.
File paths are set for AudioSet Ontology and could be changed for UrbanSound8k.
"""

def plot_signals(signals):
    """
    Plot audio signals, amplitude over time.
    Parameters: signals (dict)
    Returns: None
    """
    fig, axs = plt.subplots(2, 2, sharex=False,
                            sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    idx = 0
    for i in range(2):
        for j in range(2):
            axs[i,j].set_title(list(signals.keys())[idx])
            axs[i,j].plot(list(signals.values())[idx])
            axs[i,j].get_xaxis().set_visible(False)
            axs[i,j].get_yaxis().set_visible(False)
            idx += 1

def plot_fft(fft):
    """
    Plot fast fourier transform. (Does not calculate, use 'calc_fft')
    Parameters: fft (dict)
    Returns: None
    """
    fig, axs = plt.subplots(2, 2, sharex=False,
                            sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    idx = 0
    for i in range(2):
        for j in range(2):
            data = list(fft.values())[idx]
            Y, freq = data[0], data[1]
            axs[i,j].set_title(list(fft.keys())[idx])
            axs[i,j].plot(freq, Y)
            axs[i,j].get_xaxis().set_visible(False)
            axs[i,j].get_yaxis().set_visible(False)
            idx += 1

def plot_fbank(fbank):
    """
    Plot log filterbank energies. 
    Parameters: fbank (dict)
    Returns: None
    """
    fig, axs = plt.subplots(2, 2, sharex=False,
                            sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    idx = 0
    for i in range(2):
        for j in range(2):
            axs[i,j].set_title(list(fbank.keys())[idx])
            axs[i,j].imshow(list(fbank.values())[idx],
                            cmap='hot', interpolation='nearest')
            axs[i,j].get_xaxis().set_visible(False)
            axs[i,j].get_yaxis().set_visible(False)
            idx += 1

def plot_mfccs(mfccs):
    """
    Plot Mel Frequency cepstrum coefficients. 
    Parameters: mfccs (dict)
    Returns: None
    """
    fig, axs = plt.subplots(2, 2, sharex=False,
                            sharey=True, figsize=(20,5))
    fig.suptitle('MFCCs', size=16)
    idx = 0
    for i in range(2):
        for j in range(2):
            axs[i,j].set_title(list(mfccs.keys())[idx])
            axs[i,j].imshow(list(mfccs.values())[idx],
                            cmap='hot', interpolation='nearest')
            axs[i,j].get_xaxis().set_visible(False)
            axs[i,j].get_yaxis().set_visible(False)
            idx += 1

def plot_mels(mels):
    """
    Plot Mel Spectrograms. 
    Parameters: mels (dict)
    Returns: None
    """
    fig, axs = plt.subplots(2, 2, sharex=False,
                            sharey=True, figsize=(20,5))
    fig.suptitle('Mel Spectrograms', size=16)
    idx = 0
    for i in range(2):
        for j in range(2):
            axs[i,j].set_title(list(mels.keys())[idx])
            librosa.display.specshow(list(mels.values())[idx], x_axis='s',
                             y_axis='mel', fmax=44100, ax=axs[i,j], cmap='hot')
            axs[i,j].get_xaxis().set_visible(False)
            axs[i,j].get_yaxis().set_visible(False)
            axs[i,j].set_title(list(mels.keys())[idx])
            idx += 1  

def plot_class_dist(df, audio_path='audio/train/'):
    """ Bar graph showing average length of samples per class"""
    df.set_index('fname', inplace=True)
    for f in tqdm(df.index):
        rate, signal = wavfile.read(audio_path+f)
        df.at[f, 'length'] = signal.shape[0] / rate
    classes = list(np.unique(df.labels))
    class_dist = df.groupby(['labels'])['length'].mean().sort_values()
    df.reset_index(inplace=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title('Class Distribution', y=1.08)
    ax.barh((class_dist.index[-74:]),class_dist[-74:])
    ax.set_xlabel('Average Sample Length (seconds)')
    ax.set_ylabel('Class')
    plt.tight_layout()

def plot_audio_transforms(df, audio_path='audio/train/'):
    """ Plotting transformation process for example from each class.
    Paramters: df (pandas DataFrame) - includes column 'labels',
                                        expects column 0 to contain filename,
                                        which exists as a .wav at <audio_path>
    Returns: None 
    """
    # dict accumulators. Keys are class names, values are arrays (different stages of featurization)
    signals = {}
    fft = {}
    fbank = {}
    mfccs = {}
    mels = {} 

    classes = list(np.unique(df.labels))
    # extract features from an example
    for c in tqdm(classes):
        wav_file = df[df.labels == c].iloc[0, 0]
        signal, rate = librosa.load(audio_path + wav_file, sr=22050)
        mask = envelope(signal, rate, 0.0005)
        signal = signal[mask]
        signals[c] = signal
        fft[c] = calc_fft(signal, rate)
        fbank[c] = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
        mfccs[c] = mfcc(signal[:rate], rate, numcep=26, nfilt=26, nfft=1103).T
        mel = (librosa.feature.melspectrogram(signal, rate, n_mels=26,
                                                n_fft=1103))
        mels[c] = librosa.power_to_db(mel)
    # create plots
    plot_signals(signals)
    plot_fft(fft)
    plot_fbank(fbank)
    plot_mfccs(mfccs)
    plot_mels(mels)
    plt.tight_layout()

def envelope(y, rate, threshold):
    """ Return audio signal above threshold decibel level. Peaks are detected with rolling average windows.
    Parameters: y (array-like) - the original signal
                rate (int) - samplerate, eg. 44100 hz
                threshold (float) - dB threshold level
    Returns: Masked audio signal
    """
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/5), min_periods=1, center=True).mean()
    mask = y_mean > threshold
    return mask

def calc_fft(y, rate):
    """" Calculate the fast Fourier transform of a single signal.
    Parameters: y(array) - audio signal
                rate(int) - sample rate, eg. 44100 hz
    Returns: magnitude, freq (tuple) - 'magnitude': real discrete Fourier transformation of y
                                     - 'freq': discrete Fourier transform sample frequencies
    """"
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    magnitude = abs(np.fft.rfft(y)/n)
    return (magnitude, freq)

def process(df, noisy_df, test_df, 
            dir_train, dir_train_noisy, dir_test,
            clean_out='clean/', test_out='audio/test_roadsound/'):
    """
    Takes 3 dataframes (2 training, 1 test).
    Expects each dataframe references files in corresponding dir.
    Training data outputs to 'clean/' directory. Clear directory to re-process.
    Test data outputs to 'audio/test_roadsound/'.
    """
    if len(os.listdir(clean_out)) == 0:
        # downsample, mask and save
        for f in tqdm(df.fname):
            signal, rate = librosa.load(dir_train + f, sr=16000)
            mask = envelope(signal, rate, 0.00005)
            wavfile.write(filename=clean_out + f, rate=rate, data=signal[mask])
        for f in tqdm(noisy_df.fname):
            signal, rate = librosa.load(dir_train_noisy + f, sr=16000)
            mask = envelope(signal, rate, 0.0005)
            wavfile.write(filename=clean_out + f, rate=rate, data=signal[mask])

    if len(os.listdir(test_out)) == 0:
        # downsample and save only (no masking)
        for f in tqdm(test_df.fname):
            signal, rate = librosa.load(dir_test + f, sr=16000)
            wavfile.write(filename=test_out + f, rate=rate, data=signal)


if __name__ = '__main__':
    # Load csvs
    df = pd.read_csv('data/train/roadsound_labels.csv')
    noisy_df = pd.read_csv('data/train_noisy/roadsound_labels.csv')
    test_df = pd.read_csv('data/test/roadsound_labels.csv')
    # Define audio directories that correspond to 'fname' column in csv
    dir_train = 'audio/train/'
    dir_train_noisy = 'audio/train_noisy/'
    dir_test = 'audio/test'
    # Plot features extractions, class distribution
    plot_audio_transforms(df=df, audio_path=dir_train)
    plot_class_dist(df=df, audio_path=dir_train)
    plt.show()
    # Downsample and/or mask and save
    process(df=df,
            noisy_df=noisy_df,
            test_df=test_df,
            dir_train=dir_train,
            dir_train_noisy=dir_train_noisy,
            dir_test=dir_test)
    print('Processing complete')