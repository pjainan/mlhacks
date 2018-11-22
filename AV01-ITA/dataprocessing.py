import numpy as np
import io
import os.path as ops
import pickle
import pandas as pd

class DataHelper:
    __datafolder = "./data/"
    __testfolder = "test"
    __trainfolder = "train"
    __trainfile = "train.csv"
    __trainfile = "test.csv"

    __pickle_obj = {}

    def __init__(self):
        pass
   
    def __pickle_dump(self,pobj):
        picklefileobj = open(ops.abspath(self.__datafolder+"proj.pickle"), 'ab')
        pickle.dump(pobj, picklefileobj)
        picklefileobj.close()

    def __read_data(self, mode, path):
        rd = pd.read_csv(path)
        self.__pickle_obj[mode] = rd
        self.__pickle_dump(self.__pickle_obj)
        return rd

    def read_train_data(self):
        rd_train = (self.__read_data('train',ops.abspath(self.__datafolder+self.__trainfile)))
        print(rd_train.head())
        return rd_train
