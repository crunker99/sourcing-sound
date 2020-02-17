from tqdm import tqdm
import pandas as pd
import numpy as np

def get_single_labels(TOT='train', subset={}):
    df = pd.read_csv('data/{}/labels.csv'.format(TOT))
    df['labels'] = df['labels'].apply(lambda x: x.split(','))
    df = df[df['labels'].map(len) == 1]
    df['labels'] = df['labels'].apply(lambda x: ''.join(x))
    df.drop(['freesound_id', 'license'], 1, inplace=True)
    df = df[df['labels'].isin(subset)]
    #drop corrupted files
    for f in tqdm(df.index):
        try:
            rate, signal = wavfile.read('audio/{}/{}'.format(TOT, f))
        except ValueError:
            df.drop(f, 0, inplace=True)
    return df

if __name__ == "__main__":
    TOT = 'train'
    roadsound = {'Accelerating_and_revving_and_vroom', 'Bicycle_bell', 'Bus', 
                    'Car_passing_by' 'Motorcycle', 'Race_car_and_auto_racing', 
                    'Skateboard', 'Traffic_noise_and_roadway_noise',}
    df = get_single_labels(TOT, road_sounds)
    df.to_csv('data/{}/roadsound_labels.csv'.format(TOT))

