import os
import pandas as pd
import numpy as np
import librosa
import IPython.display as ipd
from pipeline import Pipeline


class Machine_Learning(object):
        print("Enter to Machine Learning")
        inverse_worddict=dict((v, k) for k, v in Pipeline.worddict.items())

        def __init__(self, model, weight_path, prediction_path):
            self.model=model
            self.weight_path=weight_path
            self.prediction_path=prediction_path
        
        def transfer_learning(self):
            print("loading weigth")
            return self.model.load_weights(self.weight_path)

        #PREDICTION
        def mfcc_extractor(self, file):
                signal, sample_rate = librosa.load(file)
                ipd.Audio(file)
                mfccs_features = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40)
                mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
                
                return mfccs_scaled_features

        def prediction(self):
            print("Enter to Prediction")   
            predictiondata_feataures =[]
            counter_true=0
            for root, directories, files in os.walk(self.prediction_path): 
                for file_names in files:
                    if root is not self.prediction_path:
                        r = self.mfcc_extractor(os.path.join(root,file_names))
                        rl=r.reshape(1,-1)
                        predicted_label=tuple(np.argmax(self.model.predict(rl), axis=1))
                        prediction_class = self.inverse_worddict[predicted_label[0]]
                        predictiondata_feataures.append([os.path.basename(os.path.normpath(root)),prediction_class,predicted_label[0]])
                        if os.path.basename(os.path.normpath(root))==prediction_class:
                            counter_true+=1

            predictiondata_feataures_dataframe= pd.DataFrame(predictiondata_feataures, columns=["actual data", "predicted data", "predicted label"])
            percentage= (counter_true/len(predictiondata_feataures))*100
            print("The percantage of the true predicted data: ", percentage,"%")
            print(predictiondata_feataures_dataframe)
            