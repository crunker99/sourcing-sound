import os
import numpy as np
from tensorflow.keras.models import load_model



##load ALL data
vec_type = 'mels'
data_path = os.path.join('pickles', 'urbansound_'+ vec_type + '.p')

with open(data_path, 'rb') as handle:
    data = pickle.load(handle)

X, y, folds = data[0], data[1], data[2]


if vec_type == 'mfccs':
    num_rows = 40
elif vec_type == 'mels':
    num_rows = 60 

num_columns = 174
num_channels = 1

X = X.reshape(X.shape[0], num_rows, num_columns, num_channels)

num_labels = y.shape[1]
class_weights = compute_class_weight(class_weight='balanced' , classes=np.unique(y_flat), y=y_flat )

num_epochs = 100
num_batch_size = 8

# start the timer before training.
start = datetime.now()


checkpoint = ModelCheckpoint(filepath=f'models/final_cnn.hdf5', 
                            monitor='val_acc', verbose=1, save_best_only=True,
                            save_weights_only=False)

history = model.fit(X_train, y_train, batch_size=num_batch_size,
        epochs=num_epochs, class_weight=class_weights, validation_data=(0.2), 
        callbacks=[checkpoint], verbose=1)

duration = datetime.now() - start
print("Training completed in time: ", duration)

score_test = history.history['val_acc'][-1]
print("Final Testing Accuracy: ", score_test)

best_score = max(history.history['val_acc'])
print("Best Testing Accuracy: ", best_score)
