import numpy as np
import pandas as pd
import librosa
from src.feature_extraction import call_s3


def read_audio(prep, data, trim_long_data):
    try:
        y, sr = librosa.load(data, sr=prep.sampling_rate)
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
        y = np.pad(y, (offset, prep.samples - len(y) - offset), 'reflect')
    return y


def audio_to_melspectrogram(prep, audio):
    if audio.size == 0:
        return np.zeros((prep.n_mels,prep.n_mels))
    spectrogram = librosa.feature.melspectrogram(audio, 
                                                 sr=prep.sampling_rate,
                                                 n_mels=prep.n_mels,
                                                 hop_length=prep.hop_length,
                                                 n_fft=prep.n_fft,
                                                 fmin=prep.fmin,
                                                 fmax=prep.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


def show_melspectrogram(conf, mels, title='Log-frequency power spectrogram'):
    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 
                             sr=conf.sampling_rate, hop_length=conf.hop_length,
                            fmin=conf.fmin, fmax=conf.fmax,  cmap='gist_ncar')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

def read_as_melspectrogram(prep, data, trim_long_data, debug_display=False):
    x = read_audio(prep, data, trim_long_data)
    mels = audio_to_melspectrogram(prep, x)
    if debug_display:
        IPython.display.display(IPython.display.Audio(x, rate=prep.sampling_rate))
        show_melspectrogram(prep, mels)
    return mels, x


def convert_wav_to_image(df, s3_client, bucket_name, s3_folder, sample_size=5):
    X = {} # image accumulator
    flat = {} # flat accumulator
    # audio_data = {} # audio accumulator
    for i, fname in enumerate(df.loc[:sample_size-1, 'fname']):
        data = call_s3(s3_client, bucket_name, fname, s3_folder)
        mels, aud = read_as_melspectrogram(prep, data, trim_long_data=True, debug_display=False)
        X[fname] = mels
        flat[fname] = mels.reshape(-1)
        # audio_data[fname] = aud
        print(i, ' appended ',fname)

    flat_df = pd.DataFrame(flat).T.reset_index()
    flat_df['fname'] = flat_df['index']
    flat_df = pd.merge(df, flat_df, how='inner', on='fname').drop(['index'], 1)
    # audio_df = pd.DataFrame(audio_data).T.reset_index()
    # audio_df ['fname'] = audio_df['index']
    # audio_df = pd.merge(df, audio_df, how= 'inner', on='fname').drop(['index'], 1)
    audio_df = pd.DataFrame()
    return X, flat_df, audio_df


def image_merge():
    pass

class prep:
    # Preprocessing settings
    sampling_rate = 44100
    duration = 15 # seconds to trim down if long data
    hop_length = 347*duration # to make time steps 128
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    samples = sampling_rate * duration