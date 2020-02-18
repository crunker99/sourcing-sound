import os
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


p_path = os.path.join('pickles', 'conv.p')
with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
model = load_model(config.model_path)

