import os

class Config:
    def __init__(self, mode='conv', 
                nfilt=26,
                nfeat=13, 
                nfft=512,
                rate =16000,
                n_mels=26,
                feature_type='mels'):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/5) # capture 1/10 second to make prediction
        self.n_mels = n_mels
        self.feature_type = feature_type
        self.model_path = os.path.join('models', mode + 'tt1.model')
        self.p_path = os.path.join('pickles', mode + '.p')
        self.test_p_path = os.path.join('pickles', mode + '_test.p')