import os
from tqdm import tqdm
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa
import timeit

# def plot_signals(signals):
#     fig, axs = plt.subplots(2, 4, sharex=False,
#                             sharey=True, figsize=(20,5))
#     fig.suptitle('Time Series', size=16)
#     idx = 0
#     for i in range(2):
#         for j in range(4):
#             axs[i,j].set_title(list(signals.keys())[idx])
#             axs[i,j].plot(list(signals.values())[idx])
#             axs[i,j].get_xaxis().set_visible(False)
#             axs[i,j].get_yaxis().set_visible(False)
#             idx += 1

# def plot_fft(fft):
#     fig, axs = plt.subplots(2, 4, sharex=False,
#                             sharey=True, figsize=(20,5))
#     fig.suptitle('Fourier Transforms', size=16)
#     idx = 0
#     for i in range(2):
#         for j in range(4):
#             data = list(fft.values())[idx]
#             Y, freq = data[0], data[1]
#             axs[i,j].set_title(list(fft.keys())[idx])
#             axs[i,j].plot(freq, Y)
#             axs[i,j].get_xaxis().set_visible(False)
#             axs[i,j].get_yaxis().set_visible(False)
#             idx += 1

# def plot_fbank(fbank):
#     fig, axs = plt.subplots(2, 4, sharex=False,
#                             sharey=True, figsize=(20,5))
#     fig.suptitle('Filter Bank Coefficients', size=16)
#     idx = 0
#     for i in range(2):
#         for j in range(4):
#             axs[i,j].set_title(list(fbank.keys())[idx])
#             axs[i,j].imshow(list(fbank.values())[idx],
#                             cmap='hot', interpolation='nearest')
#             axs[i,j].get_xaxis().set_visible(False)
#             axs[i,j].get_yaxis().set_visible(False)
#             idx += 1

# def plot_mfccs(mfccs):
#     fig, axs = plt.subplots(2, 4, sharex=False,
#                             sharey=True, figsize=(20,5))
#     fig.suptitle('Filter Bank Coefficients', size=16)
#     idx = 0
#     for i in range(2):
#         for j in range(4):
#             axs[i,j].set_title(list(mfccs.keys())[idx])
#             axs[i,j].imshow(list(mfccs.values())[idx],
#                             cmap='hot', interpolation='nearest')
#             axs[i,j].get_xaxis().set_visible(False)
#             axs[i,j].get_yaxis().set_visible(False)
#             idx += 1

def envelope(y, rate, threshold):
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    mask = y_mean > threshold
    return mask

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    magnitude = abs(np.fft.rfft(y)/n)
    return (magnitude, freq)


df = pd.read_csv('data/train/roadsound_labels.csv')
df.set_index('fname', inplace=True)

# for f in tqdm(df.index):
#     rate, signal = wavfile.read('audio_train/'+f)
#     df.at[f, 'length'] = signal.shape[0] / rate

# classes = list(np.unique(df.labels))
# class_dist = df.groupby(['labels'])['length'].mean().sort_values()

# #Bar graph showing average length of class samples
# fig, ax = plt.subplots(figsize=(20, 40))
# ax.set_title('Class Distribution', y=1.08)
# ax.barh((class_dist.index[-74:]),class_dist[-74:])
# ax.set_xlabel('Average Sample Length (seconds)')
# ax.set_ylabel('Class')
# # plt.show()
df.reset_index(inplace=True)


# signals = {}
# fft = {}
# fbank = {}
# mfccs = {}

# # Plotting examples of each class
# for c in tqdm(classes):
#     wav_file = df[df.labels == c].iloc[0, 0]
#     signal, rate = librosa.load('audio_train/'+wav_file, sr=22050)
#     mask = envelope(signal, rate, 0.0005)
#     signal = signal[mask]
#     signals[c] = signal
#     fft[c] = calc_fft(signal, rate)
#     fbank[c] = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
#     mfccs[c] = mfcc(signal[:rate], rate, numcep=26, nfilt=26, nfft=1103).T

# # plot_signals(signals)
# # plot_fft(fft)
# # plot_fbank(fbank)
# # plot_mfccs(mfccs)
# # plt.show()

if len(os.listdir('clean')) == 0:
    for f in tqdm(df.fname):
        signal, rate = librosa.load('audio/train/'+f, sr=16000)
        mask = envelope(signal, rate, 0.0005)
        wavfile.write(filename='clean/'+f, rate=rate, data=signal[mask])
