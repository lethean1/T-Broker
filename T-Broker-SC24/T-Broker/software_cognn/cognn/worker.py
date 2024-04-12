from queue import Queue
from multiprocessing import Process
import torch
import time
import traceback

from cognn.worker_common import ModelSummary_v2
from util.util import timestamp

class WorkerProc(Process):
    def __init__(self, model_list, pipe, param_trans_pipe, worker_id):
        super(WorkerProc, self).__init__()
        self.model_list = model_list
        self.pipe = pipe
        self.param_trans_pipe = param_trans_pipe
        self.worker_id = worker_id
        self.model_counter = 0
        
    def run(self):
        timestamp('worker', 'start')

        # Warm up CUDA and get shared cache
        torch.randn(1024, device='cuda')
        time.sleep(1)
        #torch.cuda.recv_shared_cache() # pylint: disable=no-member
        timestamp('worker', 'share_gpu_memory')
        
        while True:  # dispatch workers for task execution
            agent, model_name, para_cache_info, comp_cache_info = self.pipe.recv()
            if agent is None:
                exit()
            try:
                timestamp('worker', 'finish recv model name')
                model_summary = ModelSummary_v2(model_name, self.param_trans_pipe)
                timestamp('worker', 'import models')
                timestamp('worker', 'after importing models')

                # start doing training
                with torch.cuda.stream(model_summary.cuda_stream_for_computation):
                    timestamp('[JOB][BEGIN]', f'{model_name}-{self.worker_id}-{self.model_counter} begin')
                    output = model_summary.execute()
                    timestamp('[JOB][END]', f'{model_name}-{self.worker_id}-{self.model_counter} finished')
                    self.model_counter += 1
                    print('output: {}'.format(output))
#                    print ('Training time: {} ms'.format(output))
                    del output

                    timestamp('worker', 'send FNSH to pipe')
                    self.pipe.send('FNSH')
                    timestamp('worker', 'send FNSH to agent')
                    agent.send(b'FNSH')
                    #torch.cuda.clear_shared_cache()
            except Exception as e:
                traceback.print_tb()
            timestamp('[RESERVED]', f'{torch.cuda.max_memory_reserved()}')
            timestamp('worker_comp_thd', 'complete')
            # model_summary.reset_initialized(model_summary.model)
            torch.cuda.empty_cache()
