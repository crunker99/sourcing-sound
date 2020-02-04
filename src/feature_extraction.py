import numpy as np
import pandas as pd
import boto3
from io import BytesIO
import librosa
from botocore.exceptions import ClientError


def call_s3(s3_client, bucket_name, fname, folder='audio_train/'):
    """Call S3 instance to retrieve data from .wav file(or other format).
    Assumes file is in folder name path"""
    path = folder + fname
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=path)
    except ClientError as ex:
        if ex.response['Error']['Code'] == 'NoSuchKey':
            return dict()

    data = BytesIO(response['Body'].read()) 
    return data


def audio_vectorize(fname, data):
    """
    Analyze audio data in order to extract features for model development.

    Parameters:
    fname: (str)
    data: (_io.BytesIO)

    Returns:
    Feature dictionary
    """
    try:
        y, sr = librosa.load(data, mono=True, duration=10, offset = .5)
    except RuntimeError:
        return pd.Series()
    
    chroma_stft = np.mean(librosa.feature.chroma_stft(y, sr))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfccs = np.mean(mfccs, axis=1)

    vector_dict = {'fname':fname, 'chroma_stft': chroma_stft, 'spectral_centroid': spec_cent, 'spectral_bandwidth': spec_bw,
                  'rolloff': rolloff, 'zero_crossing_rate': zcr,}
    for i, mfcc in enumerate(mfccs):
        vector_dict['mfccs_{}'.format(i)] = mfcc
        
    return pd.Series(vector_dict)


def vector_merge(df, s3_client, bucket_name, s3_folder = 'audio_train/', sample_size=200, outpath=None):
    """
    Helper function for merging returned feature/vector dictionaries for multiple files in an S3 bucket,
    (number defined by sample_size) onto labelled dataframe, based on file names.
    """
    vectors = []
    for i, fname in enumerate(df.loc[:sample_size-1, 'fname']):
        data = call_s3(s3_client, bucket_name, fname, s3_folder)
        vec = audio_vectorize(fname, data)
        vectors.append(vec)
        print(i, ' ', fname)

    vec_df = pd.concat(vectors, axis=1, sort=True).T
    df2 = pd.merge(df, vec_df, how='inner', on='fname')
    return df2


