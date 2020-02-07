import numpy as np
import pandas as pd
from src.mels import mel_windowed, prep
import boto3
import os


def process_df(path):
    # read in labelled csv with file names
    df = pd.read_csv(path)
    # dropping unneeded columns
    df.drop(['license', 'freesound_id'], axis=1, inplace=True)
    return df

def merge_data(vectorizer, path, outpath, audpath, prep, s3_client, bucket_name, s3_folder, size=5):
    """
    Processes audio into mel spectrograms, links back with labels. 
    Creates and saves CSVs of flattened(vectorized) spectrograms, and original audio data.
    Creates and saves PIL Images of spectrograms.
    """
    df = process_df(path)

    flat_df = vectorizer(df=df,
                        batch=size,
                        prep=prep,
                        s3_client=s3_client, 
                        bucket_name=bucket_name,
                        s3_folder=s3_folder,
                        trim_long_data=True)
    # export cleaned data for easier continuous use/model training...
    flat_df.to_csv(outpath)
    # audio_df.to_csv(audpath)
    # for k, v in X.items():
    #     img = Image.fromarray(v, 'L')
    #     img.save('data/mel_images/{}.png'.format(k))


def main():
    train_path = 'data/train_labels.csv'
    test_path = 'data/test_labels.csv'
    
    train_out = 'data/train_windows_mel.csv'
    test_out = 'data/test_windows_mel.csv'

    train_audpath = 'data/train_audio_data.csv'
    test_audpath = 'data/test_audio_data.csv'

    s3_client = boto3.client('s3')
    bucket_name = 'jarednewstudy'
    train_s3_folder = 'audio_train/'
    test_s3_folder = 'audio_test/'
    
    vectorizer = mel_windowed

    clean_train = False
    if clean_train:
        size = int(input('How many train files to process? '))
        merge_data(vectorizer, train_path, train_out, train_audpath,
                    prep, s3_client, bucket_name, train_s3_folder, size=size)
    
    clean_test = True
    if clean_test:
        size = int(input('How many test files to process? '))
        merge_data(vectorizer, test_path, test_out, test_audpath,
                    prep, s3_client, bucket_name, test_s3_folder, size=size)


if __name__ == "__main__":
    main()
   ### os.system("say 'complete'")

