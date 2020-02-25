import os
import pandas as pd
import numpy as np

import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle

from src.wavhelp import WavHelper

def extract_features(fname, features='mfccs', max_pad_len=174):
    '''
    Extract audio features given a wav file path, and pads features to create uniform vector sizes,
    even with different audio lengths, so they are ready for a CNN model.
    Currently supports Mel Frequency Cepstrum Coefficients and Mel Spectrograms, using librosa library.
    Returns vector.
    '''

    try:
        signal, rate = librosa.load(fname, res_type='kaiser_fast') # default is kaiser best. Downsampling later anyways.
        if features == 'mfccs':
            vec = librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=40)
        elif features == 'mels':
            vec = librosa.feature.melspectrogram(y=signal, sr=rate, n_mels=60)
            # vec = librosa.power_to_db(vec)
        pad_width = max_pad_len - vec.shape[1]
        vec = np.pad(vec, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error while parsing file: ", fname)
        return None
        
    return vec


def examine_audio_files(metadata, audio_path='UrbanSoundDatasetSample/audio/'):
    ### THIS FUNCTION IS FOR EDA AND EXAMING AUDIO BIT DEPTH, SAMPLE RATE, AND NUMBER OF CHANNELS(STEREO/MONO).
    ### NOT NECESSARY FOR FEATURE EXTRACTION    
    wavfilehelper = WavHelper()
    audiodata = []

    for index, row in metadata.iterrows():
        
        file_name = os.path.join(audio_path, str(row["slice_file_name"]))
        
        data = wavfilehelper.get_file_props(file_name)
        audiodata.append(data)

    # Convert to dataframe
    audiodf = pd.DataFrame(audiodata, columns=['num_channels','sample_rate','bit_depth'])
    return audiodf



# paths should be updated for full dataset
metadata = pd.read_csv('UrbanSoundDatasetSample/metadata/UrbanSound8K.csv')
audio_path = 'UrbanSoundDatasetSample/audio/'


#if using a subsample
existing_files = os.listdir('UrbanSoundDatasetSample/audio')
metadata = metadata[metadata['slice_file_name'].isin(existing_files)].reset_index(drop=True)


# CNN expects similar sized data
max_pad_len = 174

features = []
vec_type = 'mfccs'

#iterating through each row, extracting features
for index, row in metasub.iterrows():
    
    file_name = os.path.join(audio_path, str(row["slice_file_name"]))
    label = row['class_name']
    vector = extract_features(fname=file_name, features=vec_type, max_pad_len=max_pad_len)
    
    features.append([vector, label])

featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
y_cat = to_categorical(le.fit_transform(y)) 

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state = 42)



### store the preprocessed data for further use
processed_data_split = (X_train, X_test, y_train, y_test)
data_path = os.path.join('pickles', 'urbansound_'+ vec_type + '.p')

with open(data_path, 'wb') as handle:
    pickle.dump(processed_data_split, handle, protocol=2)

