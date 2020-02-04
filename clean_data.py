import numpy as np
import pandas as pd
from src.feature_extraction import vector_merge
import boto3
import os



def train_df(path):
    # read in labelled csv with file names
    df = pd.read_csv(path).set_index('fname')
    # dropping unneeded columns, corrupted files
    df.drop(['license', 'freesound_id'], axis=1, inplace=True)
    df.drop(['f76181c4.wav', '77b925c2.wav', '6a1f682a.wav', 'c7db12aa.wav', '7752cc8a.wav', '1d44b0bd.wav'], inplace=True)
    df = df.reset_index()
    return df

def test_df(path):
        # read in labelled csv with file names
    df = pd.read_csv(path).set_index('fname')
    # dropping unneeded columns
    df.drop(['license', 'freesound_id'], axis=1, inplace=True)
    df = df.reset_index()
    return df

def process_train(path):
    s3_client = boto3.client('s3')
    bucket_name = 'jarednewstudy'
    s3_folder = 'audio_train/'
    df = train_df(path)
    df2 = vector_merge(df, s3_client, bucket_name, s3_folder, sample_size=5000)
    df2.to_csv('data/train_vectorized.csv')

def process_test(path):
    s3_client = boto3.client('s3')
    bucket_name = 'jarednewstudy'
    s3_folder = 'audio_test/'
    df = test_df(path)
    df2 = vector_merge(df, s3_client, bucket_name, s3_folder, sample_size=5000)
    df2.to_csv('data/train_vectorized.csv')



if __name__ == "__main__":

    train_path = 'data/train_labels.csv'
    test_path = 'data/test_labels.csv'
    
    clean_train = False
    if clean_train:
        process_train()
    
    clean_test = False
    if clean_test:
        process_test()
     
   # os.system("say 'complete'")

