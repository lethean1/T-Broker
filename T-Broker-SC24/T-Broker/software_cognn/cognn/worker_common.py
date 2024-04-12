import importlib

import torch
from enum import Enum
class WorkerMode(Enum):
    LEGACY = 1
    MMDET = 2
    TRANSFORMER = 3
    GNN = 4
### Class
class ModelSummary():
    def __init__(self, model_name, para_cache_info, comp_cache_info, param_trans_pipe):
        """ """
        self.task_name, self.data_name, self.num_layers = model_name[0], model_name[1], int(model_name[2])
        self.para_cache_size, self.para_cache_offset = para_cache_info[0], para_cache_info[1]
        self.comp_cache_size, self.comp_cache_offset = comp_cache_info[0], comp_cache_info[1]
        self.param_trans_pipe = param_trans_pipe
        self.load_model()
        self.data = None

    def execute(self): 
        return self.func(self.model, self.data)

    def reset_initialized(self, mod):
        if hasattr(mod, 'initialized'):
            mod.initialized = False
        for child in mod.children():
            self.reset_initialized(child)

    def insert_lock_hook(self, shape_summary_list): 
        """ """
        for _, _, _, mod_sublist in shape_summary_list:
            mod = mod_sublist[0]
            mod.initialized = False
            def hook_wait_for_parameter_lock(mod, input):
                if not mod.initialized:
                    complete_name = self.param_trans_pipe.recv()
                    if complete_name != mod.fullname:
                        raise Exception('Invalid complete trans')
                    mod.initialized = True
            mod.register_forward_pre_hook(hook_wait_for_parameter_lock)

    def load_model(self):
        model_module = importlib.import_module('task.' + self.task_name)
        self.model, self.func, self.shape_summary_list = model_module.import_task(self.data_name, self.num_layers)
        #_, self.data = model_module.import_model(self.data_name, self.num_layers)
        # Eliminate parameters and buffers
        self.reset_initialized(self.model)

        # Insert locks for waiting parameters and add pre_forward_hook to wait for locks
        self.insert_lock_hook(self.shape_summary_list)

        # Allocate fake memory for parameters
        self.cuda_stream_for_parameter = torch.cuda.Stream()
        self.cuda_stream_for_computation = torch.cuda.Stream()
        
        print('{} {} {}'.format(self.task_name, self.data_name, self.num_layers))   
        #with torch.cuda.stream(self.cuda_stream_for_parameter):
        #    torch.cuda.insert_shared_cache_for_computation(self.para_cache_size, self.para_cache_offset)
        
        #with torch.cuda.stream(self.cuda_stream_for_computation):
        #    torch.cuda.insert_shared_cache_for_computation(self.comp_cache_size, self.comp_cache_offset)

        with torch.cuda.stream(self.cuda_stream_for_parameter):
            for shape_list, param_list, buf_list, _ in self.shape_summary_list:
                for shape, p in zip(shape_list[:len(param_list)], param_list):
                    p.data = torch.empty(shape, device='cuda')
                for shape, b in zip(shape_list[len(param_list):], buf_list):
                    mod, key = b
                    mod._buffers[key] = torch.empty(shape, device='cuda')

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
        self.cuda_stream_for_parameter = torch.cuda.Stream()
        self.cuda_stream_for_computation = torch.cuda.Stream()
    
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
        task_name = model_cfg[0]
        print("[model summary]", "trying to import " + str(task_name), flush=True)
        model_module = importlib.import_module('task.' + task_name)
        try:
            model_module.is_transformer
        except:
            return False
        print("[model summary]", "got loader for " + str(task_name), flush=True)
        loader = \
            model_module.import_task(self.pipe, task_name)
        self.model = loader[0]
        self.func = loader[1]
        return True
    
    def execute(self):
        if self.mode == WorkerMode.LEGACY:
            return self.legacy_model_summary.execute()
        else:
            return self.func()
    
    def reset_initialized(self, mod):
        pass
