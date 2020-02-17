import collections

import numpy as np
import pandas as pd
import librosa

from src.feature_extraction import call_s3


def read_audio(prep, s3_client, bucket_name, fname, s3_folder, trim_long_data):
    data_stream = call_s3(s3_client, bucket_name, fname, s3_folder)
    try:
        y, sr = librosa.load(data_stream, sr=prep.sampling_rate)
    except(RuntimeError, TypeError):
        return np.array([])
    # trim silence
    if 0 < len(y): # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y) # trim, top_db=default(60)
    # make it unified length to conf.samples
    if len(y) > prep.samples: # long enough
        if trim_long_data:
            y = y[0:0+prep.samples]
    else: # pad blank
        padding = prep.samples - len(y)    # add padding at both ends
        offset = padding // 2
        y = np.pad(y, (offset, prep.samples - len(y) - offset), 'constant')
    return y, sr





def load_audio_windows(prep, s3_client, bucket_name, fname, s3_folder, trim_long_data):
    
    y, sr = read_audio(prep, s3_client, bucket_name, fname, s3_folder, trim_long_data)
    S = librosa.core.stft(y, n_fft=prep.n_fft,
                         hop_length=prep.hop_length, win_length=prep.window_length)
    mels = librosa.feature.melspectrogram(y=y, sr=sr, S=S,
                                            n_mels=prep.n_mels, fmin=prep.fmin, fmax=prep.fmax)

    window_size = prep.window_size
    window_hop = prep.window_hop
    #truncation
    start_frame = window_size
    end_frame = window_hop * (mels.shape[1] // window_hop)  
    windows = []
    for frame_idx in range(start_frame, end_frame, window_hop):
        # grab a slice of the spectogram at once
        win = mels[:, frame_idx-window_size:frame_idx]
        #normalize within frame
        win = np.log(win + 1e-9) 
        win -= np.mean(win)
        win /= np.std(win)
        print(win.shape)
        windows.append(win)
        assert win.shape == (prep.n_mels, prep.window_size)
        windows.append(win)

    return windows


def image_merge():
    pass

class prep:
    # Preprocessing settings
    sampling_rate = 22050
    duration = 10 # seconds to trim down if long data
    hop_length = 512 # to make time steps
    frames = 31
    fmin = 20
    fmax = 18000
    n_mels = 60
    n_fft = n_mels * 20 #Fast Fourier Transform
    samples = sampling_rate * duration
    window_length = 1024 #for STFT
    window_size = frames
    window_hop = window_size // 2 # amount to hop during snapshot
