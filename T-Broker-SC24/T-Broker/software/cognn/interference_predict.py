import argparse
import numpy as np
import os
from timeit import default_timer as timer

from sklearn.neural_network import MLPRegressor
import pickle

class InterferencePredict():
    def __init__(self, model_path):
        """ """
        self.model_path = model_path
        self.load_model()
        self.time = 0
        
    def load_model(self):
        self.model = None
        path = self.model_path
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.model = pickle.load(f)
        else:
            print("Failed to find model file: '{0}'".format(path))

    #def predict_interference(self, data):
    #    pred = self.model.predict(np.array([data]))
    #    return pred[0]
    def predict_interference(self, model0_metrics, model1_metrics):
        st = timer()
        # data: [*model1_metrics, model2_metrics]
        pred = [0, 0]
        pred[0] = self.model.predict([[*model0_metrics, *model1_metrics]])[0]
        pred[1] = self.model.predict([[*model1_metrics, *model0_metrics]])[0]
        ed = timer()
        self.time += ed - st
        return pred


    def fine_tune(self, data):
        I_train = []
        O_train = []
        for b in data.keys():
            I_train += data[b][0]
            O_train += data[b][1]
        I_train = np.array(I_train)
        O_train = np.array(O_train)
        self.model.set_params(max_iter=10)
        self.model.fit(I_train,O_train)
