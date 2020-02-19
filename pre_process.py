from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.io import wavfile

def get_single_labels(dir, subset:set = {}):
    """"
    Selects only samples with one label, removes unneeded columns, 
    and drops corrupted samples.
    Parameters: dir : str
                    A directory which contains a file called 'labels.csv'

                subset : array-like, default {}
                    Select only samples whose labels are in the set 
    Returns:    df: pandas DataFrame
    """"
    df = pd.read_csv('data/{}/labels.csv'.format(dir))
    df.set_index('fname', inplace=True)
    df['labels'] = df['labels'].apply(lambda x: x.split(','))
    df = df[df['labels'].map(len) == 1]
    df['labels'] = df['labels'].apply(lambda x: ''.join(x))
    df.drop(['freesound_id', 'license'], 1, inplace=True)
    df = df[df['labels'].isin(subset)]
    #drop corrupted files
    for f in tqdm(df.index):
        try:
            rate, signal = wavfile.read('audio/{}/{}'.format(dir, f))
        except ValueError:
            df.drop(f, 0, inplace=True)
    df.reset_index(inplace=True)
    return df

if __name__ == "__main__":
    roadsound = {'Bicycle_bell', 'Bus', 
                 'Motorcycle', 'Race_car_and_auto_racing', 
                    'Skateboard',}

    train_df = get_single_labels(dir='train', subset=roadsound)
    train_df.to_csv('data/train/roadsound_labels.csv')

    train_noisy_df = get_single_labels(dir='train_noisy', subset=roadsound)
    train__noisy_df.to_csv('data/train_noisy/roadsound_labels.csv')

    test_df = get_single_labels(dir='test', subset=roadsound)
    test_df.to_csv('data/test/roadsound_labels.csv')
