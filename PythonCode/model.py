from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam

class Model():
    print("Creating Model...")
    def __init__(self):
        pass

    def create_model():
        
        model= Sequential()
        model.add(Flatten(input_shape=(40, )))
        model.add(Dense(300, Activation('relu')))
        model.add(Dropout(0.2))
        model.add(Dense(300, Activation('relu')))
        model.add(Dropout(0.2))
        model.add(Dense(14, Activation('softmax')))
        model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=Adam())
        return model