import os
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.io import wavfile
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten
from tensorflow.keras.layers import LSTM, TimeDistributed
from tensorflow.keras.layers import Dropout, SpatialDropout2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pickle
from python_speech_features import mfcc
import librosa
from librosa.feature import melspectrogram

from cfg import Config
from buildfeats import build_rand_feat

def get_conv_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu',
                    strides=(1,1), padding='same', input_shape=input_shape))    
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1,1),padding='same')) 
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1,1),padding='same'))
    # model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1), padding='same'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    return model

def get_recurrent_model():
    #shape of data for RNN is (n, time, feat) 
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model


config = Config()

cur_df = pd.read_csv('data/train/roadsound_labels.csv', index_col=0)
noisy_df = pd.read_csv('data/train_noisy/roadsound_labels.csv', index_col=0)
df = pd.concat([cur_df, noisy_df], sort=True)
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('clean/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate
df = df[df.length > config.step/rate]


classes = list(np.unique(df.labels))
class_dist = df.groupby(['labels'])['length'].mean()
# n_samples = 2 * int(df['length'].sum() / 0.1) # 40 * total length of audio
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)

df, test_df, _, _ = train_test_split(df, df.labels)

if config.mode == 'conv':
    X, y = build_rand_feat(config, df, 'train')
    X_test, y_test = build_rand_feat(config, test_df, 'val')
    y_flat = np.argmax(y, axis=1) # create an array of integer labels
    input_shape = (X.shape[1], X.shape[2])
    model = get_conv_model()

elif config.mode == 'time':
    print('mode is time(recurrent)')
    X, y = build_rand_feat(df, 'train')
    X_test, y_test = build_rand_feat(test_df, 'val')
    y_flat = np.argmax(y, axis=1) # create an array of integer labels
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()

with open(config.p_path, 'wb') as handle:
    pickle.dump(config, handle, protocol=2)

class_weight = compute_class_weight('balanced',
                                    np.unique(y_flat),
                                    y_flat)

n_epochs = 10
batch_size = 32

checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=1)

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=2, 
                        batch_size=batch_size, write_graph=True, 
                        write_grads=True, write_images=True)

model.fit(X, y, epochs=n_epochs, batch_size=batch_size,
            shuffle=True, class_weight=class_weight,
            validation_data =(X_test, y_test) , callbacks=[checkpoint, tensorboard])

#if best model, save to .model_path
model.save(config.model_path)
#save all models anyway
saved_model_path = "./models/{}epochs_{}.h5".format(n_epochs, datetime.now().strftime("%Y%m%d")) # _%H%M%S 
model.save(saved_model_path)

