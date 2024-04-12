from queue import Queue
from multiprocessing import Process
import torch
import time
import gc

from cognn.worker_common import ModelSummary_v2 
from util.util import timestamp

class ModelProc(Process):
    def __init__(self, model_name, pipe, worker_id, model_id,is_combomc):
        super(ModelProc, self).__init__()
        self.model_name = model_name
        self.pipe = pipe
        self.worker_id = worker_id
        self.model_id = model_id
        self.is_combomc = is_combomc
    def run(self):
        timestamp('[MODEL_PROC]', 'start')
        print(torch.cuda.memory_summary())
        model_summary = ModelSummary_v2(self.model_name, self.pipe, self.is_combomc)
        timestamp('worker', 'import models')
        timestamp('worker', 'after importing models')

        # start doing training
        #with torch.cuda.stream(model_summary.cuda_stream_for_computation):
        timestamp('[JOB][BEGIN]', f'{str(self.model_name)}-{self.worker_id}-{self.model_id} begin')
        output = model_summary.execute()
        print('output-{}: {}'.format(self.model_name, output))
        timestamp('[JOB][END]', f'{str(self.model_name)}-{self.worker_id}-{self.model_id} finished')
#        print ('Training time: {} ms'.format(output))
        del model_summary
        del output
        

class WorkerProc(Process):
    def __init__(self, model_list, pipe, pipe_memory_threshold, worker_id, is_combomc):
        super(WorkerProc, self).__init__()
        self.model_list = model_list
        self.pipe = pipe
        self.pipe_memory_threshold = pipe_memory_threshold
        self.worker_id = worker_id
        self.model_counter = 0
        self.is_combomc = is_combomc
        
    def run(self):
        timestamp('worker', 'start')

        # Warm up CUDA 
        #torch.randn(1024, device='cuda')
        time.sleep(1)
        while True:  # dispatch workers for task execution
            agent, model_name = self.pipe.recv()
            if agent is None:
                exit()
            timestamp('worker', f'recv {model_name}')
            #gc.collect()
            #torch.cuda.empty_cache()
            #gc.collect()
            #torch.cuda.empty_cache()
            #print(torch.cuda.memory_summary())
            #model_summary = ModelSummary_v2(model_name, self.pipe_memory_threshold)
            #timestamp('worker', 'import models')
            #timestamp('worker', 'after importing models')

            ## start doing training
            ##with torch.cuda.stream(model_summary.cuda_stream_for_computation):
            #timestamp('[JOB][BEGIN]', f'{str(model_name)} begin')
            #output = model_summary.execute()
            #print('output-{}: {}'.format(model_name, output))
            #timestamp('[JOB][END]', f'{str(model_name)} finished')
#           # print ('Training time: {} ms'.format(output))
            #del model_summary
            #del output

            #gc.collect()
            #torch.cuda.empty_cache()
            #gc.collect()
            #torch.cuda.empty_cache()
            model_proc = ModelProc(model_name, self.pipe_memory_threshold, self.worker_id, self.model_counter, self.is_combomc)
            self.model_counter += 1
            model_proc.start()
            model_proc.join()
            timestamp('worker', 'send FNSH to pipe')
            self.pipe.send('FNSH')
            timestamp('worker', 'send FNSH to agent')
            if agent != 0:
                agent.send(b'FNSH')

            timestamp('worker_comp_thd', 'complete')
