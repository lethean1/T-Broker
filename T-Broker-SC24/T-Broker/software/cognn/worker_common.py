import importlib

import torch

from util.util import timestamp
from enum import Enum 
class WorkerMode(Enum):
    LEGACY = 1
    MMDET = 2
    TRANSFORMER = 3
    GNN = 4

### Class
class ModelSummary():
    def __init__(self, model_name):
        """ """
        task_name = model_cfg[0]
        if 'num_layers' not in task_name:
            return False
        model_module = importlib.import_module('task.GNN_base')

        self.func = model_module.import_func(task_name)
        
        self.task_name, self.data_name, self.num_layers = model_name[0], model_name[1], int(model_name[2])
        self.load_model()

    def execute(self): 
        return self.func(self.model, self.data)

    def load_model(self):
        model_module = importlib.import_module('task.' + self.task_name)
        self.model, self.func, _ = model_module.import_task(self.data_name, self.num_layers)
        _, self.data = model_module.import_model(self.data_name, self.num_layers)
        
        self.cuda_stream_for_computation = torch.cuda.Stream()
        
        print('{} {} {}'.format(self.task_name, self.data_name, self.num_layers))   

class ModelSummary_v2():
    def __init__(self, model_cfg, pipe,is_combomc=False):
        self.pipe = pipe
        if self.try_init_gnn(model_cfg):
            self.mode = WorkerMode.GNN
        elif self.try_init_mmdet(model_cfg):
            self.mode = WorkerMode.MMDET 
        elif self.try_init_transformer(model_cfg,is_combomc):
            self.mode = WorkerMode.TRANSFORMER
        else:
            self.mode = WorkerMode.LEGACY
            self.legacy_model_summary = ModelSummary(model_cfg)
            self.model = self.legacy_model_summary.model
    
    def try_init_gnn(self, model_cfg):
        task_name = model_cfg[0]
        if 'num_layers' not in task_name:
            return False
        model_module = importlib.import_module('task.GNN_base')

        self.func = model_module.import_func(task_name)
        return True

    def try_init_mmdet(self, model_cfg):
        task_name = model_cfg[0]
        model_module = importlib.import_module('task.' + task_name)
        try:
            model_module.import_mmdet_task
        except:
            return False
        self.func \
            = model_module.import_mmdet_task(model_cfg[0], model_cfg[1], self.pipe)
        return True
    def try_init_transformer(self, model_cfg, is_combomc):
        timestamp('worker', 'begin init transformer')
        task_name = model_cfg[0]
        model_module = importlib.import_module('task.' + task_name)
        timestamp('worker', 'finish import module')
        try:
            model_module.import_transformer_task
        except:
            timestamp('worker', 'init transformer exception')
            return False
        loader = \
            model_module.import_transformer_task(self.pipe, task_name, is_combomc)
        self.model = loader.model
        self.func = loader.do_train
        timestamp('worker', 'finish init transformer')
        return True
    
    def execute(self):
        if self.mode == WorkerMode.LEGACY:
            return self.legacy_model_summary.execute()
        else:
            return self.func()

