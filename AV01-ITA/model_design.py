import dataprocessing as dp
import numpy as np
import keras

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
        self.dh.fill_arrays('test')
        self.dh.fill_arrays('train')
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
        mo = self._model
        mo.layers.append(keras.layers.Flatten(input_shape=(28,28)))
        mo.layers.append(keras.layers.Dense(128,activation = keras.activations.relu))
        mo.layers.append(keras.layers.Dense(128,activation = keras.activations.tanh))
        mo.layers.append(keras.layers.Dense(10,activation = keras.activations.softmax))
        mo.compile(optimizer=keras.optimizers.Adam(), loss= keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        return mo

    def Train_The_Model(self):
        mo = self.ModelDefinition()
        mo.fit(self.train_data,self.train_labels,epochs=5)
        #mo.predict_classes()