import os
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.io import wavfile
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten
from tensorflow.keras.layers import Dropout, SpatialDropout2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint
from python_speech_features import mfcc
import librosa
from librosa.feature import melspectrogram
from cfg import Config
# from buildfeats import build_rand_feat

def check_data():
    if os.path.isfile(config.p_path):
        print('Loading existing data for {} model'.format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp 
    else:
        return None

def build_rand_feat(df, split):
    tmp = check_data()
    if not tmp:
        tmp = Config()
        tmp.data = [None, None, None, None]
    if split == 'train' and not tmp.data[0] is None:
            return tmp.data[0], tmp.data[1]
    elif split == 'test' and not tmp.data[2] is None:
            return tmp.data[2], tmp.data[3]
    config.data = [None, None, None, None]
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
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
        elif config.feature_type = 'raw':
            X_sample = sample
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample)
        y.append(classes.index(rand_class)) # encoding integer values for classes
    config.min = _min
    config.max = _max
    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y = to_categorical(y)
    if split == 'train':
        config.data[0], config.data[1] = (X, y)
    elif split == 'test':
        config.data[2], config.data[3] = (X, y)

    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
    return X, y

def get_conv_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu',
                    strides=(1,1), padding='same', input_shape=input_shape))    
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1,1),padding='same')) 
    model.add(SpatialDropout2D(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1),padding='same'))
    # model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1), padding='same'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    return model


df = pd.read_csv('data/train/roadsound_labels.csv', index_col=0)
df.set_index('fname', inplace=True)
for f in df.index:
    rate, signal = wavfile.read('clean/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate


classes = list(np.unique(df.labels))
class_dist = df.groupby(['labels'])['length'].mean()
n_samples = 2 * int(df['length'].sum() / 0.1) # 40 * total length of audio
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)

config = Config()

df, test_df, _ , _ = train_test_split(df, df.labels)

if config.mode == 'conv':
    X, y = build_rand_feat(df, 'train')
    X_test, y_test = build_rand_feat(test_df, 'test')
    y_flat = np.argmax(y, axis=1) # create an array of integer labels
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model()

class_weight = compute_class_weight('balanced',
                                    np.unique(y_flat),
                                    y_flat)

checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=1)

model.fit(X, y, epochs=20, batch_size=16,
            shuffle=True, class_weight=class_weight,
            validation_data =(X_test, y_test) , callbacks=[checkpoint])

#if best model, save to .model_path
model.save(config.model_path)
#save all models anyway
saved_model_path = "./models/10epochs_{}.h5".format(datetime.now().strftime("%Y%m%d")) # _%H%M%S 
model.save(saved_model_path)

