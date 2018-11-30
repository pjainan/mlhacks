import dataprocessing as dp
import numpy as np

class Model:
    dh = dp.DataHelper()
    data_store = []
    train = []
    test = [] 
    train_data = []
    train_labels = []
    test_data = [] 
    test_labels = []

    def __init__(self):
        self.ProcessData()
        self.data_store = self.ShowDataStore()
    
    def ProcessData(self):
        self.dh.fill_arrays('test')
        self.dh.fill_arrays('train')
        self.dh.get_data('train')
        self.dh.get_data('test')    
    
    def ShowDataStore(self):
        return self.dh.show_pickle()

    def Format_Train_Data(self):
        self.train_data = np.array(self.dh.pickle_extract("train_images"))