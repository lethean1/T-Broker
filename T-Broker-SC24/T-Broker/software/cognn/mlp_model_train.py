import numpy as np
import math
import random
import csv
import argparse
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor

class MLPModel():
    def init(self, num=34, hidden=(5,5)):
        model = MLPRegressor(hidden_layer_sizes=(5,5),activation='relu',alpha=0.1,batch_size=4, solver='adam',learning_rate_init=1e-3, warm_start=True, max_iter=7000, early_stopping=True,random_state=70)
        # fake fit to obtain weights
        model.fit([[0 for i in range(num)] for k in range(1000)],[0 for i in range(1000)])
        for ii in range(len(model.coefs_)):
            dd = model.coefs_[ii].tolist()
            weight = []
            for d in dd:
                d2 = np.random.uniform(0,1,len(d))
                map(lambda x: x*math.sqrt(2/len(d2)), d2)
                weight.append(d2)
            model.coefs_[ii] = np.array(weight)
        for ii in range(len(model.intercepts_)):
            dd = model.intercepts_[ii].tolist()
            weight = []
            for d in dd:
                d2 = 0.0
                # d2 = np.random.uniform(0,1,1)[0]*math.sqrt(2)
                weight.append(d2)
            model.intercepts_[ii] = np.array(weight)
        return model

    def train(self, dataset):
        data = dataset.data
        I_train = []
        O_train = []
        for b in data:
            I_train.append(b[0])
            O_train.append(b[1])        
        I_train = np.array(I_train)
        O_train = np.array(O_train).ravel()

        self.model = self.init()
        self.model.fit(I_train, O_train)

        O_train_pred = self.model.predict(I_train)
        train_loss = mean_absolute_error(O_train, O_train_pred)
        print("\n Training Loss: ", train_loss)
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
                self.data.append(([row[i] for i in range(34)], [row[34]]))
                
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train mlp model from collected metrics')
    parser.add_argument('--out', help="output model's file name", default=None)
    args = parser.parse_args()
    mlp_model = MLPModel()
    dataset = MyDataset()
    dataset.load('data.csv')
    # train with whole dataset
    print("Begin training")
    mlp_model.train(dataset)
    # save
    print("Saving trained model")
    model.save(args.out)

