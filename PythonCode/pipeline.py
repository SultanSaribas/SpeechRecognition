import os
import pandas as pd
import numpy as np
import IPython.display as ipd
import tensorflow as tf
from tensorflow import keras
import librosa


class Pipeline(tf.keras.utils.Sequence):
    worddict = {
            "ac": 0,
            "asagi": 1,
            "baslat": 2,
            "devam": 3,
            "dur": 4,
            "evet": 5,
            "geri":6,
            "hayir": 7,
            "ileri": 8,
            "iptal": 9,
            "kapa":10,
            "sag":11,
            "sol": 12,
            "yukari":13

            }

    def __init__(self, path):
        self.shuffle=True
        self.paths=path
        
    def mfcc_extractor(self,file):
        signal, sample_rate = librosa.load(file)
        ipd.Audio(file)
        mfccs_features = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        
        return mfccs_scaled_features

    def __getitem__(self, index):
        #GET MFCC FETAURES FOR EACH PATH(AUDIO)
        data = self.mfcc_extractor(self.paths[index])/10
        #GET THE LABEL FOR EACH PATH(AUDIO)
        labelY=os.path.basename(os.path.dirname(self.paths[index]))
        #IMPORT DATA TO DATAFRAME
        extracted_feataures=[]
        extracted_feataures.append([data,labelY])
        extracted_feataures_dataframe= pd.DataFrame(extracted_feataures, columns=["melspectrogram", "word"])
        #GET DATA AS A LIST
        X=np.array(extracted_feataures_dataframe['melspectrogram'].tolist())
        #TURN LABELS INTO BINARY VALUES
        y = np.empty((0,14), dtype=int)
        y_class = keras.utils.to_categorical(self.worddict[labelY], num_classes=14)
        y_data = []
        y_data.append(y_class)
        y = np.concatenate((y, np.array(y_data)))
        return X, y


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.paths)


    def __len__(self):
        return len(self.paths)
        #pass