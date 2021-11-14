import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from sklearn import preprocessing

class Preprocessing:
    def __init__(self, log):
        self.data = []
        self.train = []
        self.train_y = []
        self.test = []
        self.test_y = []
        self.result = []
        self.log = pd.read_csv(log)
        self.colNames = self.log.columns
        self.log = self.log.to_numpy()


    def get_data(self):
        row = []
        for l in range(300):
            #center
            center = self.fetch(self.log[l][0])
            #left
            left = self.fetch(self.log[l][1])
            #right
            right = self.fetch(self.log[l][2])

            row.append(center)
            row.append(left)
            row.append(right)
            #add rest of elements
            for k in range(3, len(self.log[l]) - 1):
                self.result.append(self.log[l][k])

            
            self.data.append(row)
            self.result.append(self.log[l][-1])
            row = []

        print(np.shape(self.data[0][0]))
        print(self.data[0][0])

        self.seperate_data()

        return self.data
        

    def seperate_data(self):
        self.train_data = self.data[:int(len(self.data) * .7)]
        self.train_y = self.result[:int(len(self.result) * .7)]
        self.test_data = self.data[int(len(self.data) * .7):]
        self.test_y = self.result[int(len(self.data) * .7):]
        

    def fetch(self, name):
        data = Image.open("data/" + name.replace(" ", ""))
        data = ImageOps.grayscale(data)
        data = np.asarray(data)
        data = preprocessing.normalize(data)
        return data


    def preprocess(self, image):
        #normalize value
        pass

    
    