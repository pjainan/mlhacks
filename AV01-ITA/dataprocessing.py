import numpy as np
import io
import os
import os.path as ops
import pickle
import pandas as pd
# import matplotlib.pyplot as plt
from skimage.io import imread

class DataHelper:
    __datafolder = "./data/"
    __testfolder = "test"
    __trainfolder = "train"
    __trainfile = "train"
    __trainfile = "test"
    __file_ext = ".csv"

    __pickle_obj = {}
    __unpickle_obj = {}

    def __init__(self):
        pass
   
    def pickle_dump(self):
        picklefileobj = open(ops.abspath(self.__datafolder+"proj.pickle"), 'ab')
        pickle.dump(self.__pickle_obj, picklefileobj)
        picklefileobj.close()

    def pickle_extract(self):
        picklefileobj = open(ops.abspath(self.__datafolder+"proj.pickle"), 'rb')
        __unpickle_obj = pickle.load(picklefileobj)
        for keys in __unpickle_obj:
            print(keys)
        picklefileobj.close()

    def __read_data(self, mode, path):
        rd = pd.read_csv(path)
        self.__pickle_obj[mode] = rd
        return rd

    def get_data(self, mode):
        data = self.__read_data(mode,ops.abspath(self.__datafolder+mode+self.__file_ext))
        return data

    def fill_arrays(self, mode):
        imgs = []
        image_folder = ops.abspath(self.__datafolder+mode+"/")
        for dirpath, dirs, images in os.walk(image_folder):
            for img in images:
                imgs.append(imread(ops.join(image_folder, img)))
            print("captured images data : " + mode)
            self.__pickle_obj[mode+"_images"] = imgs
                            