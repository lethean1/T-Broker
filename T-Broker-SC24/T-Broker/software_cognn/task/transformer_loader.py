import torch
from torch.profiler import profile, ProfilerActivity
class TransformerLoader:
    def __init__(self, fn,  name, raw_args):
        self.generator = fn(raw_args) 
        self.model = next(self.generator)
        do_train = next(self.generator)
        self.name = name
        
        def do_train_with_pipe(*args):
           # with profile(
           #     activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
           #     with_stack=True,
           #     profile_memory=True,
           # ) as prof:
            do_train() 
            #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=100))
            #with open('time.txt', 'a') as f:
            #    f.write('[model]\n')
            #    f.write(self.name)
            #    f.write('\n')
            #    f.write('[time]\n')
            #    f.write(str(prof.key_averages().table(row_limit=1000)))
            #    f.write('\n')
        self.do_train = do_train_with_pipe
    
    

    
    
    