import numpy as np
import math
import random
import csv
import argparse
import json
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
import datetime
import pickle

import matplotlib.pyplot as plt

class MLPModel():
    def init(self, num=26, hidden=(8, 4, 4)):
        model = MLPRegressor(verbose=True, alpha=0.01, learning_rate="adaptive", activation="tanh", solver="lbfgs", hidden_layer_sizes=hidden, max_iter=15000, random_state=42, early_stopping=True, tol=1e-6, n_iter_no_change=100)
        return model

    def train(self, dataset):
        data = dataset.data
        I_train = []
        O_train = []
        for b in data:
            I_train.append(b[0])
            O_train.append(b[1])        
        I_train = np.array(I_train)
        O_train = np.array(O_train)

        self.model = self.init()
        self.model.fit(I_train, O_train)

        print("========================================================")
        print(len(O_train))

        O_train_pred = self.model.predict(I_train)
        # O_train_pred_manual = [1.22 for i in range(500)]
        O_train_pred_manual = O_train_pred
        # print(np.array([O_train_pred_manual, O_train]).transpose()[:10])
        # print(np.array([O_train_pred_manual, O_train]).transpose()[-10:])
        train_loss = mean_absolute_error(O_train, O_train_pred_manual)
        print("\n Training Loss: ", train_loss)
        # print("Training precents: ", (O_train - O_train_pred_manual)/O_train)
        #for i in range(len(O_train)):
        #    print(O_train[i])
        return self.model
    
    def predict(self, dataset):
        data = dataset.data
        I_test = []
        O_test = []
        for b in data:
            I_test.append(b[0])
            O_test.append(b[1])        
        I_test = np.array(I_test)
        O_test = np.array(O_test)

        O_test_pred = self.model.predict(I_test)
        arr = np.array([O_test_pred, O_test]).transpose()
        # for i in range(min(100, len(arr))):
        #     print(arr[i])
        # for i in range(min(100, len(arr))):
        #     print(arr[-i])
        test_loss = mean_absolute_error(O_test, O_test_pred)
        print("\n Test Loss: ", test_loss)
        rel_error = np.abs((O_test - O_test_pred)/O_test)
        print("precent mean: ", rel_error.mean())
        print(rel_error)

        # export data
        # rel_error = np.sort(rel_error)
        with open("combomc_test_accuracy.json", "w") as f:
            data = []
            for i in range(len(rel_error)):
                data.append(float(rel_error[i]))
            json.dump({"data": data}, f, indent=2)
        
        return self.model
    
    def fine_tune(self, path, data):
        I_train = []
        O_train = []
        for b in data.keys():
            I_train += data[b][0]
            O_train += data[b][1]
        I_train = np.array(I_train)
        O_train = np.array(O_train)

        model = load(path)
        model.set_params(max_iter=10)
        model.fit(I_train,O_train)

        O_train_pred = model.predict(I_train)
        train_loss = mean_absolute_error(O_train, O_train_pred)
        print("\n Training Loss:{}\nmape:{} ".format(train_loss,mape))
        return model
        
    def load(self, path):
        self.model = None
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        return self.model

    def save(self, name=None):
        if name is None:
            name = self.__class__.__name__ + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.pkl'
        print("Save model into: {0}".format(name))
        pickle.dump(self.model, open(name,"wb"))
    
class MyDataset():
    def __init__(self):
        self.data = []
    
    def load(self, data_path):
        flag = 0
        with open(data_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if flag == 0:
                    flag = 1
                    continue
                row = pd.to_numeric(row)
                self.data.append(([row[i] for i in range(26)], row[26]))
                
        
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Train mlp model from collected metrics')
#     parser.add_argument('--out', help="output model's file name", default=None)
#     args = parser.parse_args()
#     mlp_model = MLPModel()
#     dataset = MyDataset()
#     dataset.load('train_data_combomc.csv')
#     # train with whole dataset
#     print("Begin training")
#     mlp_model.train(dataset)
#     # save
#     if args.out is not None:
#         print("Saving trained model")
#         mlp_model.save(args.out)
#     if True:
#         print("Begin testing")
#         test_dataset = MyDataset()
#         test_dataset.load('test_data_combomc.csv')
#         mlp_model.predict(test_dataset)

mlp_model = MLPModel()
mlp_model.load("model_0806.pkl")
test_dataset = MyDataset()
test_dataset.load('test_data_combomc.csv')
mlp_model.predict(test_dataset)
