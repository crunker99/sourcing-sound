import numpy as np
import pandas as pd
from librosa.core import stft
from librosa.feature import melspectrogram
import librosa
from src.feature_extraction import call_s3
import matplotlib.pyplot as plt

class prep:
    # Preprocessing settings
    sampling_rate = 22050
    duration = 10 # seconds to trim down if long data
    hop_length = 512 # to make time steps
    frames = 86
    fmin = 20
    fmax = sampling_rate
    n_mels = 60
    n_fft = n_mels * 20 #Fast Fourier Transform
    samples = sampling_rate * duration
    window_length = 1024 #for STFT
    window_size = (sampling_rate // hop_length) * 5   #  a second window
    window_hop = window_size // 4   # amount to hop during snapshot
    
    # 1 frame ~= 1/43 second (sample rate / hop length)
    # 21.5 frames = 0.5 seconds


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


def load_audio_windows(prep, label, fname, s3_client, bucket_name, s3_folder, trim_long_data):
    try:
        y, sr = read_audio(prep, s3_client, bucket_name, fname, s3_folder, trim_long_data)
    except ValueError:
        return [np.zeros((1, prep.n_mels * int(prep.window_size)))]
    S = stft(y, n_fft=prep.n_fft,
                         hop_length=prep.hop_length, win_length=prep.window_length)
    mels = melspectrogram(y=y, sr=sr, S=S,
                        n_mels=prep.n_mels, fmin=prep.fmin, fmax=prep.fmax)

    window_size = int(prep.window_size)
    window_hop = int(prep.window_hop)
    #truncation method
    start_frame = window_size
    end_frame = mels.shape[1] - window_hop 
    windows = []
    for frame_idx in range(start_frame, end_frame, window_hop):
        # grab a slice of the spectogram at once
        win = mels[:, frame_idx-window_size:frame_idx]
        #normalize within frame
        # win = librosa.core.power_to_db(win, top_db=80)
        win = np.log(win + 1e-9) 
        win -= np.mean(win)
        win /= np.std(win)
#         print(win.shape)
        win = np.hstack(win)
        win = np.append(win, label)
        win = np.append(win, fname)
        windows.append(win)
#         assert win.shape == (prep.n_mels, prep.window_size)
    return windows

def mel_windowed(df, batch, prep, s3_client, bucket_name, s3_folder, trim_long_data=True):
    acc = []
    labs = []
    i=0
    for row in df.iloc[:batch,:].iterrows():
        print(i, " ", row[1][1])
        i += 1
        windows = load_audio_windows(prep=prep, 
                                    label=row[1][1],
                                    fname=row[1][0],
                                    s3_client=s3_client,
                                    bucket_name=bucket_name, 
                                    s3_folder=s3_folder, 
                                    trim_long_data=trim_long_data)
    #     windows = flatten(windows)
        for win in windows:
            acc.append(win[:-1])
            labs.append(win[-1])
            
    Meldf = pd.DataFrame(acc, labs)
    Meldf.columns = [*Meldf.columns[:-1], 'labels']
    return Meldf


def show_melspectrogram(prep, mels, title='Log-frequency power spectrogram', cmap='gist_ncar', i=0):
    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 
                             sr=prep.sampling_rate, hop_length=prep.hop_length,
                            fmin=prep.fmin, fmax=10000,  cmap=cmap)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.savefig('img/mel_win{}.png'.format(i), dpi=256)
    plt.show()