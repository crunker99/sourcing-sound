import numpy as np
import pandas as pd
from src.feature_extraction import vector_merge
from src.wav2img import convert_wav_to_image
import boto3
import os


# def train_df(path):
#     # read in labelled csv with file names
#     df = pd.read_csv(path).set_index('fname')
#     # dropping unneeded columns, corrupted files
#     df.drop(['license', 'freesound_id'], axis=1, inplace=True)
#     # df.drop(['f76181c4.wav', '77b925c2.wav', '6a1f682a.wav', 'c7db12aa.wav', '7752cc8a.wav', '1d44b0bd.wav'], inplace=True)
#     df = df.reset_index()
#     return df

# def test_df(path):
#     # read in labelled csv with file names
#     df = pd.read_csv(path).set_index('fname')
#     # dropping unneeded columns
#     df.drop(['license', 'freesound_id'], axis=1, inplace=True)
#     df = df.reset_index()
#     return df

# def process_test(path, outpath='data/test_vectorized.csv', s3_client, bucket_name, s3_folder, size=5):
#     df = test_df(path)
#     df2 = vector_merge(df, s3_client, bucket_name, s3_folder, sample_size=size)
#     df2.to_csv(outpath)

def process_df(path):
    # read in labelled csv with file names
    df = pd.read_csv(path).set_index('fname')
    # dropping unneeded columns
    df.drop(['license', 'freesound_id'], axis=1, inplace=True)
    df = df.reset_index()
    return df

def merge_data(vectorizer, path, outpath, s3_client, bucket_name, s3_folder, size=5):
    df = process_df(path)
    X, flat_df, audio_data = vectorizer(df, s3_client, bucket_name, s3_folder, sample_size=size)
    flat_df.to_csv(outpath)


if __name__ == "__main__":
    train_path = 'data/train_labels.csv'
    test_path = 'data/test_labels.csv'
    
    train_out = 'data/train_mels_vec.csv'
    test_out = 'data/test_mels_vec.csv'

    s3_client = boto3.client('s3')
    bucket_name = 'jarednewstudy'
    train_s3_folder = 'audio_train/'
    test_s3_folder = 'audio_test/'
    
    # vectorizater = vector_merge()
    vectorizer = convert_wav_to_image


    clean_train = True
    if clean_train:
        size = int(input('How many train files to process? '))
        merge_data(vectorizer, train_path, train_out, s3_client, bucket_name, train_s3_folder, size=size)
    
    clean_test = False
    if clean_test:
        size = int(input('How many test files to process? '))
        merge_data(vectorizer, test_path, test_out, s3_client, bucket_name, test_s3_folder, size=size)

   ### os.system("say 'complete'")

