import numpy as np
import pandas as pd
from src.feature_extraction import vector_merge


# Loading labeled data
df = pd.read_csv('data/train_labels.csv').set_index('fname')
# dropping unneeded columns, corrupted files
df.drop(['license', 'freesound_id'], axis=1, inplace=True)
df.drop(['f76181c4.wav', '77b925c2.wav', '6a1f682a.wav', 'c7db12aa.wav', '7752cc8a.wav', '1d44b0bd.wav'], inplace=True)
df = df.reset_index()

