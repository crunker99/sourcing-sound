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
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import LeaveOneGroupOut
from sklearn import metrics 


def get_conv_model():
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
    # model.summary()
    return model



# Retrieve data from pickle file. 
# From ubs_process.py, will be a 3-item tuple(X, y(categorical matrix), folds)

vec_type = 'mfccs'
data_path = os.path.join('pickles', 'urbansound_'+ vec_type + '.p')

with open(data_path, 'rb') as handle:
    data = pickle.load(handle)

X, y, folds = data[0], data[1], data[2]


# Pre-specify global variables for model
# num_rows as specified by number of mfccs or mels. 
# Columns expected to be same as <max_pad_len> in ubs_process

num_rows = 40
num_columns = 174
num_channels = 1

X = X.reshape(X.shape[0], num_rows, num_columns, num_channels)

num_labels = y.shape[1]
filter_size = 2

# # Pre training scores
# model = get_conv()
##### score = model.evaluate(X_test, y_test, verbose=1) 
# accuracy = 100*score[1]
# print("Pre-training accuracy: %.4f%%" % accuracy)


### TRAINING

# user specified number of epochs
num_epochs = int(input('Enter number of epochs: '))
# num_epochs = 72
num_batch_size = 256

# start the timer before training. This will include all the fold durations
start = datetime.now()

### Cross validation. Fold indices pre-specified

fold_accuracies = {}

logo = LeaveOneGroupOut()

for train_idx, test_idx in logo.split(X, y, folds):

    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    fold = folds[test_idx][0]

    model = get_conv_model()

    #create checkpoint to save best model 
    checkpoint = ModelCheckpoint(filepath=f'models/basic_cnn_fold{fold}.hdf5', 
                            monitor='val_acc', verbose=1, save_best_only=True,
                            save_weights_only=False)

    # put the different runs into a tensorboard log directory
    log_dir = f"logs/fit/fold{fold}_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=2, 
                        batch_size=num_batch_size, write_graph=True, 
                        write_grads=True, write_images=True)

    start_fold = datetime.now()

    history = model.fit(X_train, y_train, batch_size=num_batch_size,
            epochs=num_epochs, validation_data=(X_test, y_test), 
            callbacks=[checkpoint, tensorboard], verbose=1)
    
    duration_fold = datetime.now() - start_fold
    print("Fold training completed in time: ", duration_fold)

    # Evaluating the model on the training and validating set
    # score_train = model.evaluate(X_train, y_train, verbose=0)
    # print("Training Accuracy: ", score_train[1])

    score_test = history.history['val_acc'][-1]
    print("Final Testing Accuracy: ", score_test)

    best_score = max(history.history['val_acc'])
    print("Best Testing Accuracy: ", best_score)

    fold_accuracies[fold] = best_score

    clear_session()



### Review results of total training
duration = datetime.now() - start
print("Training completed in time: ", duration)

# compute average accuracy

for k, v in sorted(fold_accuracies.items()):
    print(f'Fold {k}:    accuracy = {v}')

avg_score = np.mean([v for v in fold_accuracies.values()])
print('Average Accuracy: ', avg_score)

# # Evaluating the model on the training and testing set
# score = model.evaluate(X_train, y_train, verbose=0)
# print("Training Accuracy: ", score[1])

# score = model.evaluate(X_test, y_test, verbose=0)
# print("Testing Accuracy: ", score[1])

