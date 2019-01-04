import dataprocessing as dp
import numpy as np
import keras
import matplotlib.pyplot as plt
import sklearn
import pickle
from sklearn.model_selection import train_test_split
import model_configurations as mc

class Model:
    dh = dp.DataHelper()
    md = mc.ModelDefinition()
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

    _histories = []
    total_iterations = 1
    history_pkl = './data/mo_hist/fashion_mnist-history.pkl'
    predictions = {}

    def __init__(self, data_processing_mode):
        if data_processing_mode == "1":
           self.ProcessData()
           self.data_store = self.ShowDataStore()
        elif data_processing_mode == "0":
            self.Format_Data()
            self.Train_The_Model(0)
        elif data_processing_mode == "2":
        # Mode to evaluate model on validation and test data set
            self.Format_Data()
            self.Evaluate_on_Test(0)
        elif data_processing_mode == "3":
        # Mode to generate the predictions from the training and validations 
            self.Estimate_the_Output()
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
        

        self.train_data = self.train[0:50000,:,:,:]
        self.train_labels = self.train_target[0:50000]

        # self.val_data = self.train[40000:50000,:,:,:]
        # self.val_labels = self.train_target[40000:50000]
        
        self.test_data = self.train[50000:60000,:,:,:]
        self.test_labels = self.train_target[50000:60000]

    def ModelDefinition(self, model_config):
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(28,28,4)))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.convolutional.Conv2D(filters=model_config["layer1filters"], kernel_size = model_config["layer1kernel"], data_format='channels_last',  padding=model_config["layer1pad"],  activation='relu')) 
        model.add(keras.layers.MaxPooling2D(pool_size=model_config["layer1pool"]))
        model.add(keras.layers.Dropout(model_config["drop1"]))
        
        model.add(keras.layers.convolutional.Conv2D(filters=model_config["layer2filters"], kernel_size = model_config["layer2kernel"], padding=model_config["layer2pad"],  activation='relu')) 
        model.add(keras.layers.MaxPooling2D(pool_size=model_config["layer2pool"]))
        model.add(keras.layers.Dropout(model_config["drop2"]))
        
        model.add(keras.layers.Flatten())
        
        model.add(keras.layers.Dense(model_config["layer1dense"], activation='relu'))
        model.add(keras.layers.Dropout(model_config["drop3"]))
        model.add(keras.layers.Dense(model_config["layer2dense"], activation='relu'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Dense(10, activation='softmax'))
        model.compile(optimizer=keras.optimizers.Adam(), loss= keras.losses.categorical_crossentropy, metrics=['accuracy'])
        return model

    def Train_The_Model(self, model_index):
        histories = []
        model_config = self.md.model_definitions[model_index]
        for i in range(0, model_config["iterations"]):
            mo = self.ModelDefinition(model_config)
            print('Model Summary for the model : %i and iteration - %i : ' % (model_index, i ))
            print(mo.summary())
            print("Running for model# [%i] - iteration (%i): " % (model_index, i))
            filepath = self.dh.generate_iteration_model_file("fmnist_model_%i_iteration_%i" % (model_index, i))
            checkpoint = keras.callbacks.ModelCheckpoint(filepath,monitor='val_loss', save_best_only=True,mode='min')
            train_x, val_x, train_y, val_y = sklearn.model_selection.train_test_split(self.train_data, self.train_labels, test_size=0.20, random_state=1001)
            
            hist = mo.fit(train_x,train_y, epochs=self._epochs, batch_size=self._batch_size, validation_data=(val_x, val_y), callbacks=[checkpoint])
            histories.append(hist.history)

        with open(self.history_pkl, 'wb') as f:
            pickle.dump(histories, f) 
        
        print("Training Evaluation for model# [%i] - iteration (%i): " % (model_index, i))
        self.Evaluate_on_Train()

    def Evaluate_on_Train(self):
        fileobj = open(self.history_pkl,'rb')
        histories = pickle.load(fileobj)
        fileobj.close()
        print('Training: \t%0.8f loss / %0.8f acc' % (self.Get_Training_History_Average(histories, 'loss'), self.Get_Training_History_Average(histories, 'acc')))
        print('Validation: \t%0.8f loss / %0.8f acc' % (self.Get_Training_History_Average(histories, 'val_loss'), self.Get_Training_History_Average(histories, 'val_acc')))

    def Get_Training_History_Average(self,histories,attrib):
        tmp = []
        for history in histories:
           tmp.append(history[attrib][np.argmin(history['val_loss'])])
        return np.mean(tmp)

    def Evaluate_on_Test(self, model_index):
        test_loss = []
        test_accuracy = []
        self.predictions = {}
        # Grab the test data
        self.test = np.array(self.dh.pickle_extract("test_images"))
        self.test = self.test/255.0
        model_config = self.md.model_definitions[model_index]
        for i in range(0, model_config["iterations"]):
            temp_cnn = self.ModelDefinition(model_config)
            temp_cnn.load_weights(self.dh.get_iteration_model_file("fmnist_model_%i_iteration_%i" % (model_index, i)))
            score = temp_cnn.evaluate(self.test_data, self.test_labels, verbose = 0)
            test_loss.append(score[0])
            test_accuracy.append(score[1])
            print("Test with model# [%i] - iteration (%i): %0.8f loss / %0.8f accuracy " %(model_index, i, score[0], score[1]))
        
            print("Capture predictions for model# [%i] - iteration (%i) Test Data. " %(model_index, i))
            p = temp_cnn.predict(self.test, verbose=0)
            p = np.argmax(p,axis=1)
            self.predictions[("model%iiteration%i" % (model_index, i))] = p

        
        # plt.imshow(self.train_data[1])
        # plt.show(block=True)
        # print(self.class_names[self.train_labels[1]])
        print('\n Avg test loss/accuracy : \t %0.4f /  %0.4f' % (np.mean(test_loss), np.mean(test_accuracy)))

    def Estimate_the_Output(self):
        self.dh.load_predictions(self.predictions)
            
