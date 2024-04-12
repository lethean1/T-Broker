import json
from timeit import default_timer as timer
from cognn.task_info_table import task_info_table
from math import floor, ceil
with open("task_recompute_strategy.json") as f:
    task_recompute_strategy = json.load(f)

with open("module_recompute_time_info.json") as f:
    module_recompute_time_info = json.load(f)

for key in task_recompute_strategy:
    peak = task_info_table[key]['peak_memory']
    max_threshold = max(task_recompute_strategy[key], key=lambda x: float(x))
    val = task_recompute_strategy[key][max_threshold]
    # added: 230806 20:50
    
    max_threshold = float(max_threshold)
    max_threshold += 0.5
    if max_threshold < peak:
        #print(key, task_recompute_strategy[key][f'{max_threshold-0.5:.1f}'] , peak, max_threshold)
        pass
    while max_threshold < peak:
        #task_info_table[key][f'{max_threshold:.1f}'] = []
        task_recompute_strategy[key][f'{max_threshold:.1f}'] = val
        max_threshold += 0.5

for key in task_recompute_strategy:
    peak = task_info_table[key]['peak_memory']
    threshold = task_info_table[key]['memory_threshold']
    while threshold < peak:
        if str(threshold) not in task_recompute_strategy[key]:
            print(key, threshold)
        threshold += 0.5


class RecomputationPredict():
    def __init__(self):
        """ """
        # todo
        self.time = 0
        pass
        
    def predict_recomputation(self, memory, task_name, time):
        st = timer()
        
        if memory >= ceil(task_info_table[task_name]['peak_memory']*2)/2:
            return time
        memory_str = f"{memory:.1f}"
        if task_name not in task_recompute_strategy:
            print("[RECOMPUTE] ", task_name, "Not found", flush=True)
            return 1e10
        if memory_str not in task_recompute_strategy[task_name]:
            print("[RECOMPUTE] ", task_name, memory_str, "Not found", flush=True)
            print("[RECOMPUTE] ", task_recompute_strategy[task_name].keys(), flush=True)
            return 1e10
        recompute_cost = 0
        for module in task_recompute_strategy[task_name][memory_str]:
            recompute_cost += module_recompute_time_info[task_name][module][0]
        recompute_cost /= 1e9
    
        ed = timer()
        self.time += ed - st
        return time + recompute_cost
