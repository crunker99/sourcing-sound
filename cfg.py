import os

class Config:
    def __init__(self, mode='conv', 
                nfilt=26,
                nfeat=13, 
                nfft=512,
                rate =16000,
                n_mels=60,
                feature_type='raw',
                pca=False):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/5) #note: rate = 1 second
        self.n_mels = n_mels
        self.feature_type = feature_type
        self.model_path = os.path.join('models', mode + feature_type + 'tt1.model')
        self.p_path = os.path.join('pickles', mode + feature_type +'.p')
        self.test_p_path = os.path.join('pickles', mode + feature_type + '_test.p')
