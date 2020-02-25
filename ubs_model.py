import os
import numpy as np
from datetime import datetime 
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from sklearn import metrics 


def get_conv():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()
    return model



#retrieve data from pickle file. From ubs_process.py, will be a 4-item tuple(train-test split)

vec_type = 'mfccs'
data_path = os.path.join('pickles', 'urbansound_'+ vec_type + '.p')

with open(data_path, 'rb') as handle:
    data = pickle.load(handle)

X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]



#pre-specify global variables for model
## num_rows as specified by number of mfccs or mels. Columns expected to be same as <max_pad_len> in ubs_process

num_rows = 40
num_columns = 174
num_channels = 1

X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns, num_channels)
X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns, num_channels)

num_labels = y_train.shape[1]
filter_size = 2

model = get_conv()


score = model.evaluate(X_test, y_test, verbose=1)

accuracy = 100*score[1]
print("Pre-training accuracy: %.4f%%" % accuracy)


### TRAINING

num_epochs = int(input('Enter number of epohchs:    '))
num_batch_size = 128

# num_epochs = 72
# num_batch_size = 256

checkpoint = ModelCheckpoint(filepath='models/weights.best.basic_cnn.hdf5', 
                               verbose=1, save_best_only=True)

log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=2, 
                        batch_size=num_batch_size, write_graph=True, 
                        write_grads=True, write_images=True)
start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, 
            validation_data=(X_test, y_test), callbacks=[checkpoint, tensorboard], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)

# Evaluating the model on the training and testing set
score = model.evaluate(X_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])

