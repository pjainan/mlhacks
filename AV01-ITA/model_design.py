import dataprocessing as dp
import numpy as np
import keras
import matplotlib.pyplot as plt

class Model:
    dh = dp.DataHelper()
    data_store = []
    train = []
    test = [] 
    train_data = []
    train_labels = []
    test_data = [] 
    test_labels = []
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    _model = keras.Sequential()

    def __init__(self, data_processing_mode):
        if data_processing_mode == "1":
           self.ProcessData()
           self.data_store = self.ShowDataStore()
        elif data_processing_mode == "0":
            self.Format_Data()
            self.Train_The_Model()
        else:
            print("Mention correct commandline parameter.")
            
    
    def ProcessData(self):
        self.dh.fill_arrays('test',60001,70000)
        self.dh.fill_arrays('train',1,60000)
        self.dh.get_data('train')
        self.dh.get_data('test')    
    
    def ShowDataStore(self):
        return self.dh.show_pickle()

    def Format_Data(self):
        self.train_data = np.array(self.dh.pickle_extract("train_images"))
        self.train_data = self.train_data/255.0
        self.test_data = np.array(self.dh.pickle_extract("test_images"))
        self.test_data = self.test_data/255.0
        self.train_labels = np.array(self.dh.pickle_extract("train_labels"))[:,1]

    def ModelDefinition(self):
        # mo = self._model
        # mo.add(keras.layers.Flatten(input_shape=(28,28)))
        # mo.add(keras.layers.Dense(64,activation = keras.activations.relu))
        # # mo.add(keras.layers.Dense(128,activation = keras.activations.tanh))
        # mo.add(keras.layers.Dense(10,activation = keras.activations.softmax))
        # mo.compile(optimizer=keras.optimizers.Adam(), loss= keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        # return mo
        model = self._model
        # model.add(keras.layers.InputLayer(input_shape=(64, 28,28, 1)))
        model.add(keras.layers.convolutional.Conv2D(filters=64, kernel_size=3, data_format='channels_last', input_shape=(28,28,4), padding='same',  activation='relu')) 
        model.add(keras.layers.MaxPooling2D(pool_size=2))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=2))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(10, activation='softmax'))
        model.compile(optimizer=keras.optimizers.Adam(), loss= keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        return model

    def Train_The_Model(self):
        mo = self.ModelDefinition()
        print('Model Summary for the current for the processed model')
        print(mo.summary())
        mo.fit(self.train_data,self.train_labels,epochs=5)
        # mo.fit(self.train_data,self.train_labels,epochs=5, batch_size=100)
        # mo.predict
        # plt.imshow(self.train_data[1])
        # plt.show(block=True)
        # print(self.class_names[self.train_labels[1]])
       