import dataprocessing as dp
import numpy as np
import keras
import matplotlib.pyplot as plt
import sklearn
import pickle
from sklearn.model_selection import train_test_split

class Model:
    dh = dp.DataHelper()
    data_store = []
    train = []
    test = [] 
    train_target = []
    test_target = []

    val_data = []
    val_labels = []
    train_data = []
    train_labels = []
    test_data = [] 
    test_labels = []
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    _classes = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
    _num_classes = len(_classes)

    _batch_size = 128
    _epochs = 10
    _histories = []
    total_iterations = 1

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
        self.train = np.array(self.dh.pickle_extract("train_images"))
        self.train_target = keras.utils.np_utils.to_categorical((np.array(self.dh.pickle_extract("train_labels"))[:,1]),num_classes=self._num_classes)
        # self.train_target = np.array(self.dh.pickle_extract("train_labels"))[:,1]
        self.train = self.train/255.0
        
        self.test = np.array(self.dh.pickle_extract("test_images"))
        self.test = self.test/255.0

        self.train_data = self.train[0:50000,:,:,:]
        self.train_labels = self.train_target[0:50000]

        # self.val_data = self.train[40000:50000,:,:,:]
        # self.val_labels = self.train_target[40000:50000]
        
        self.test_data = self.train[50000:60000,:,:,:]
        self.test_labels = self.train_target[50000:60000]

        print(self.test_data.shape)
       


    def ModelDefinition(self):
        model = self._model
        model.add(keras.layers.InputLayer(input_shape=(28,28,4)))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.convolutional.Conv2D(filters=24, kernel_size = 3, data_format='channels_last',  padding='same',  activation='relu')) 
        model.add(keras.layers.MaxPooling2D(pool_size=2))
        model.add(keras.layers.Dropout(0.3))
        
        model.add(keras.layers.Conv2D(filters=64, kernel_size = 3, padding='same', activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=2))
        model.add(keras.layers.Dropout(0.4))

        #model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
        #model.add(keras.layers.MaxPooling2D(pool_size=2))
        #model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.Flatten())
        
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        
        model.add(keras.layers.Dense(128, activation='relu'))

        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Dense(10, activation='softmax'))
        
        model.compile(optimizer=keras.optimizers.Adam(), loss= keras.losses.categorical_crossentropy, metrics=['accuracy'])
        return model

    def Train_The_Model(self):
        mo = self.ModelDefinition()
        print('Model Summary for the current for the processed model')
        print(mo.summary())

        for i in range(0, self.total_iterations):
            print("Running for iteration: %i" % i)
            filepath = self.dh.generate_iteration_model_file("fmnist_mo1", str(i))
            checkpoint = keras.callbacks.ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,mode='min')
            train_x, val_x, train_y, val_y = sklearn.model_selection.train_test_split(self.train_data, self.train_labels, test_size=0.15, random_state=1001)
            
            hist = mo.fit(train_x,train_y, epochs=self._epochs, batch_size=self._batch_size, validation_data=(val_x, val_y), callbacks=[checkpoint])
            self._histories.append(hist)

        with open('./data/mo_hist/fashion_mnist-history.pkl', 'wb') as f:
            pickle.dump(self._histories, f)    
        # # Evaluate the model on test set
        # score = mo.evaluate(self.test_data, self.test_labels, verbose=0)
        # # Print test accuracy
        # print('\n', 'Test accuracy:', score[1])
        # # mo.predict
        # # plt.imshow(self.train_data[1])
        # # plt.show(block=True)
        # # print(self.class_names[self.train_labels[1]])
       
