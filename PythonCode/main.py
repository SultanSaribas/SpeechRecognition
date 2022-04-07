from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from machine_learning import Machine_Learning
from pipeline import Pipeline
import matplotlib as plt
from model import Model
import pandas as pd
import numpy as np
import os

def main():

    root_path="C:/Users/sarib/Desktop/Gohm/SpeechRecognition/database"
    def get_paths(get_root_path, pathlist):
        for root, directories, files in os.walk(get_root_path):
            for file_names in files:
                if root is not get_root_path:
                    pathlist.append(os.path.join(root,file_names))

    allpaths=[]
    #GET ALL PATHS INTO ARRAY
    get_paths(get_root_path=root_path, pathlist=allpaths)
    #ASSIGN THE ARRAY INTO DATAFRAME TO BE ABLE TO SPLIT
    dataframe= pd.DataFrame(allpaths, columns=["paths"])
    np.random.shuffle(dataframe.values)
    #SPLIT THE DATA 80% FOR TRAINING, 10% FOR VALIDATION, 10% FOR TESTING DATA
    train_data = dataframe[:21188]
    valid_data = dataframe[21188:23836:]
    test_data= dataframe[23836:]
    #PUT THEM ON SEPERATE LISTS
    training_set=np.array(train_data['paths'].tolist())
    validation_set=np.array(valid_data['paths'].tolist())
    test_set=np.array(test_data['paths'].tolist())

    #DATA GENERATION WITH KERAS.UTILS.SEQUENCE
    train_prefit = Pipeline(path=training_set)
    valid_prefit = Pipeline(path=validation_set)
    test_prefit= Pipeline(path=test_set)

    #CREATE MODEL
    model= Model.create_model()
    print("Training the data!...")
    checkpointpath="checkpoint/cp.ckpt"
    checkpointer = ModelCheckpoint(checkpointpath, monitor='val_accuracy',
                                verbose=2, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=2)
    callbackList= [checkpointer, early_stop]
    history=model.fit(train_prefit, validation_data=valid_prefit, epochs=200, callbacks=[callbackList], verbose=2)

    #EVALUATE
    print("Test Score: ", Model.evaluate(test_prefit))
    #PLOT THE ACCURACY VALUES
    print("Plotting the accuracy values")
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label = 'test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title("1")

      ####ADD SECOND DATASET####
    second_data_paths=get_paths("C:/Users/sarib/Desktop/Gohm/SpeechRecognition/RecordsFromDataset")
    second_data_prefit=Pipeline(second_data_paths)
    history2=model.fit(second_data_prefit, validation_data=valid_prefit, epochs=50, callbacks=[early_stop], verbose=2)
    #EVALUATE
    print("Test Score: ", model.evaluate(test_prefit))
    #PLOT THE ACCURACY VALUES
    print("Plotting the accuracy values")
    plt.plot(history2.history['accuracy'], label='train')
    plt.plot(history2.history['val_accuracy'], label = 'test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title("1")

    #PREDICTION
    Machine_Learning(model, checkpointpath,"C:/Users/sarib/Desktop/Gohm/SpeechRecognition/Records")

if __name__== "__main__" :
    main()