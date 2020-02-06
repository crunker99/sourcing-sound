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


"""
def audio_to_melspectrogram(prep, y):
    if y.size == 0:
        return np.zeros((prep.n_mels,prep.n_mels))
    mels = librosa.feature.melspectrogram(y, 
                                                 sr=prep.sampling_rate,
                                                 n_mels=prep.n_mels,
                                                 hop_length=prep.hop_length,
                                                 n_fft=prep.n_fft,
                                                 fmin=prep.fmin,
                                                 fmax=prep.fmax)
    # spectrogram = librosa.power_to_db(spectrogram)
    mels = mels.astype(np.float32)
    return mels


def sample_windows(length, frame_samples, window_frames, overlap=0.5, start=0):
  

    # Split @samples into a number of windows of samples
    with length @frame_samples * @window_frames

    ws = frame_samples * window_frames
    while start < length:
        end = min(start + ws, length)
        yield start, end
        start += (ws * (1-overlap))


def show_melspectrogram(prep, mels, title='Log-frequency power spectrogram'):
    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 
                             sr=conf.sampling_rate, hop_length=conf.hop_length,
                            fmin=conf.fmin, fmax=conf.fmax,  cmap='gist_ncar')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

def read_as_melspectrogram(prep, fname, window_frames,start_time=None,
                             normalize='meanstd', trim_long_data, debug_display=False):
    y = read_audio(prep, fname, trim_long_data)
    mels = audio_to_melspectrogram(prep, y)
    assert mels.shape[0] == prep.n_mels, mels.shape

    if start_time is None:
        # Sample a window in time randomly
        min_start = max(0, mels.shape[1] - window_frames)
        if min_start == 0:
            start = 0
        else:
            start = np.random.randint(0, min_start)
    else:
        start = int(start_time * (sample_rate / hop_length))

    end = start + window_frames
    mels = mels[:, start:end]

    # Normalize the window
    if mels.shape[1] > 0:
        if normalize == 'max':
            mels /= (np.max(mels) + 1e-9)
            mels = librosa.core.power_to_db(mels, top_db=80)
        elif normalize == 'meanstd':
            mels = librosa.core.power_to_db(mels, top_db=80)
            mels -= np.mean(mels)
            mels /= ( np.std(mels) + 1e-9)
        else:
            mels = librosa.core.power_to_db(mels, top_db=80, ref=0.0)
    else:
        print('Warning: Sample {} with start {} has 0 length'.format(sample, start_time))

    # Pad to standard size
    if window_frames is None:
        padded = mels
    else:
        padded = np.full((n_mels, window_frames), 0.0, dtype=float)
        inp = mels[:, 0:min(window_frames, mels.shape[1])]
        padded[:, 0:inp.shape[1]] = inp
    
    if debug_display:
        IPython.display.display(IPython.display.Audio(x, rate=prep.sampling_rate))
        show_melspectrogram(prep, mels)

    # add channel dimension
    data = np.expand_dims(padded, -1)
    return data


Sample = collections.namedtuple('Sample', 'start end fold slice_file_name')


def load_windows(fname, prep, overlap, start=0):
    sample_rate = prep.sampling_rate
    frame_samples = prep.hop_length
    window_frames = prep.frames

    windows = []

    duration = sample.end - sample.start
    length = int(sample_rate * duration)

    for win in sample_windows(length, frame_samples, window_frames, overlap=overlap, start=start):
        chunk = Sample(start=win[0]/sample_rate,
                       end=win[1]/sample_rate,
                       fold=sample.fold,
                       slice_file_name=fname)    
        d = read_as_melspectrogram(prep, )
        windows.append(d)

    return windows

"""





def load_audio_windows(prep, s3_client, bucket_name, fname, s3_folder, trim_long_data):
    
    y, sr = read_audio(prep, s3_client, bucket_name, fname, s3_folder, trim_long_data)
    S = librosa.core.stft(y, n_fft=prep.n_fft,
                         hop_length=prep.hop_length, win_length=prep.window_length)
    mels = librosa.feature.melspectrogram(y=y, sr=sr, S=S,
                                            n_mels=prep.n_mels, fmin=prep.fmin, fmax=prep.fmax)
    #using truncation
    start_frame = prep.window_size
    end_frame = prep.window_hop * math.floor(float(prep.frames) / prep.window_hop)
    windows = []
    for frame_idx in range(start_frame, end_frame, window_hop):
        # grab a slice of the spectogram at once
        win = mels[:, frame_idx - prep.window_size:frame_idx]

        win = np.log(win + 1e-9) 
        win -= np.mean(win)
        win /= np.std(win)

        assert win.shape == (prep.n_mels, prep.window_size)
        windows.append(win)

    return windows

"""def convert_wav_to_image(df, s3_client, bucket_name, s3_folder, outpath, sample_size=5):
    loader = read_audio
    X = {} # image accumulator
    flat = {} # flat accumulator
    # audio_data = {} # audio accumulator
    for i, fname in enumerate(df.index[:sample_size]):
        windows = load_windows(fname, prep, overlap=0.5)
        inputs = np.stack(windows)



        data = read_as_melspectrogram(prep, data, window_frames, start_time=None,
                                            normalize='meanstd'
                                            trim_long_data=False, debug_display=False)
        
        X[fname] = mels
        flat[fname] = mels.reshape(-1)
        # audio_data[fname] = aud
        if i % 10 == 0:
            print(i, ' appended ',fname)

    flat_df = pd.DataFrame(data=flat.values(), index=flat.keys())
    flat_df = pd.merge(df, flat_df, left_index=True, right_index=True)
    # audio_df = pd.DataFrame(audio_data).T.reset_index()
    # audio_df ['fname'] = audio_df['index']
    # audio_df = pd.merge(df, audio_df, how= 'inner', on='fname').drop(['index'], 1)
    audio_df = pd.DataFrame()
    return X, flat_df"""


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
    window_length = 31 
    window_hop = 0.5 # amount to hop during snapshot