import numpy as np
import io
import os
import os.path as ops
import pickle
import pandas as pd
import matplotlib.pyplot as plt
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

    def pickle_extract(self, key):
        picklefileobj = open(ops.abspath(self.__datafolder+"proj.pickle"), 'rb')
        __unpickle_obj = pickle.load(picklefileobj)
        picklefileobj.close()
        return __unpickle_obj[key]

    def __read_data(self, mode, path):
        rd = pd.read_csv(path)
        self.pickle_modify(mode+"_labels",rd)
        return rd

    def get_data(self, mode):
        data = self.__read_data(mode,ops.abspath(self.__datafolder+mode+self.__file_ext))
        return data

    def fill_arrays(self, mode, start_index, end_index):
        imgs = []
        image_folder = ops.abspath(self.__datafolder+mode+"/")
        for dirpath, dirs, images in os.walk(image_folder):
            for img in range(start_index,end_index+1):
                # imgs.append(np.array(imread(ops.join(image_folder, img))[:,:,0]))
                # im = np.array(imread(ops.join(image_folder, img),as_gray=True))
                im = np.array(imread(ops.join(image_folder, str(img)+".png")))
                imgs.append(im)
            self.pickle_modify(mode+"_images",imgs)


    #extract a particular object from the pickle file.                            
    def pickle_extract_key(self, extract_key):        
        picklefileobj = open(ops.abspath(self.__datafolder+"proj.pickle"), 'rb')
        __unpickle_obj = pickle.load(picklefileobj)
        ret_obj = __unpickle_obj[extract_key]
        picklefileobj.close()
        return ret_obj

    def pickle_modify(self, extract_key, new_obj):
        __unpickle_obj = dict()
        picklefileobj = open(ops.abspath(self.__datafolder+"proj.pickle"), 'rb')
        try:
            __unpickle_obj  = pickle.load(picklefileobj)
            open(ops.abspath(self.__datafolder+"proj.pickle"), 'w').close()
            picklefileobj.close()
        except EOFError as error:
            print("return empty pickle")
        #clear content of the files
        if extract_key in __unpickle_obj.keys():
            #update the object contents
            __unpickle_obj[extract_key] = new_obj
        else:
            #add new the object contents
            __unpickle_obj[extract_key] = new_obj
        #update the pickle contents
        picklefileobj = open(ops.abspath(self.__datafolder+"proj.pickle"), 'ab')
        pickle.dump(__unpickle_obj, picklefileobj)
        #close the file
        picklefileobj.close()

    def show_pickle(self):
        __unpickle_obj = dict()
        try:
            picklefileobj = open(ops.abspath(self.__datafolder+"proj.pickle"), 'rb')
            __unpickle_obj  = pickle.load(picklefileobj)
            picklefileobj.close()
        except EOFError as error:
            print("return empty shw_pickle")
        for key in __unpickle_obj.keys():
            print("key: " + key + " has following shape : " + str(np.array(__unpickle_obj[key]).shape))
        return __unpickle_obj
