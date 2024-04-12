import torch
from cognn.interference_predict import InterferencePredict
import json
from cognn.task_info import TaskInfo
from cognn.recomputation_predict import RecomputationPredict
from math import ceil, floor
from util.util import timestamp
from timeit import default_timer as timer
DEVICE_MEMORY = 32
### Class
#class Schedule():
#    def __init__(self, job_list):
#        """ """
#        self.job_list = job_list
#
#    def get_co_job(self):
#        co_job = self.job_list.pop(0)
#        return co_job, self.job_list
schedule_time = 0
def get_schedule_time():
    global schedule_time
    return schedule_time
def time(func):
    def wrapper(*args, **kwargs):
        global schedule_time
        st = timer()
        ret = func(*args, **kwargs)
        ed = timer()
        schedule_time += ed - st
        return ret
    return wrapper
class ScheduleBase:
    def __init__(self, job_list):
        self.full_job_list = job_list
        self.time_st = timer()
    def get_cur_job_list(self):
        time = timer() - self.time_st
        return [x for x in self.full_job_list if int(x[3]) < time]
    def only_not_ready_job(self):
        return len(self.full_job_list) != 0 and len(self.get_cur_job_list()) == 0
    def finished(self):
        return len(self.full_job_list) == 0

def get_first_job_wrapper(func):
    def wrapper(self):
        self.job_list = self.get_cur_job_list()
        se_job, job_list = func(self)
        self.full_job_list.remove(se_job)
        return se_job, self.full_job_list
    return wrapper
def get_co_job_wrapper(func):
    def wrapper(self, job):
        self.job_list = self.get_cur_job_list()
        st, job_list, sm1, sm2 = func(self, job)
        if st in self.full_job_list:
            self.full_job_list.remove(st)
            st.append(sm2)
        if len(job) != 5:
            job.append(sm1)
        return job, st, self.full_job_list, sm1, sm2
    return wrapper

        
class Schedule0(ScheduleBase):
    def __init__(self, job_list, model_path):
        """ """
        super(Schedule0, self).__init__(job_list)
        self.job_list = job_list
        self.predict_model = InterferencePredict(model_path)
        self.recomputation = RecomputationPredict() 
    @get_first_job_wrapper
    def get_first_job(self):
        min_training_time = 1e9
        for i in range(len(self.job_list)):
            task_name = self.job_list[i][1]
            task_info = TaskInfo(task_name)
            training_time = task_info.get_training_time()
            if training_time < min_training_time:
                min_training_time = training_time
                se_job = self.job_list[i]
        
        self.job_list.remove(se_job)
        return se_job, self.job_list
            
               
    @get_co_job_wrapper
    def get_co_job(self, job):
        
        task_name = job[1]
        task_info = TaskInfo(task_name)
        peak_memory = task_info.get_peak_memory()
        metrics = task_info.get_metrics()
        training_time = task_info.get_training_time()
        throughput = 0

        if len(self.job_list) == 0:
            return None, self.job_list, peak_memory, 0
        selected_task = None
        selected_memory_1 = peak_memory
        selected_memory_2 = 0
        for i in range(len(self.job_list)):
            se_task_name = self.job_list[i][1]
            se_task_info = TaskInfo(se_task_name)
            se_peak_memory = se_task_info.get_peak_memory()
            se_metrics = se_task_info.get_metrics()
            se_training_time = se_task_info.get_training_time()
          
            data = [*metrics, *se_metrics]
            infer_factor = self.predict_model.predict_interference(metrics, se_metrics)
            timestamp('[INF_FACT]', str((task_name, se_task_name, infer_factor)))
          
            if se_peak_memory + peak_memory < DEVICE_MEMORY:
                #t_1 = infer_factor[0] * training_time # todo 需要敏感度系数吗？看结果误差 wsq
                t_2 = infer_factor[1] * se_training_time
                if 1/t_2 > throughput:
                    throughput = 1/t_2
                    selected_task = self.job_list[i]
                    selected_memory_1 = peak_memory
                    selected_memory_2 = se_peak_memory
            else:
                compressed_memory = se_peak_memory + peak_memory - DEVICE_MEMORY
                for j in range(-1, ceil(compressed_memory)*2+2):
                    memory_1 = floor(peak_memory * 2 - j ) / 2
                    memory_2 = floor(se_peak_memory * 2 - (ceil(compressed_memory)*2 - j))/2

                    timestamp("[VERBOSE_SCHEDULE]", "Trying {}-{} {}-{}".format(task_name, memory_1, se_task_name, memory_2))

                    t_r_1 = self.recomputation.predict_recomputation(memory_1, task_name, training_time)
                    t_r_2 = self.recomputation.predict_recomputation(memory_2, se_task_name, se_training_time)
                    
                    timestamp("[VERBOSE_SCHEDULE]", "{}-{} {}-{} {} {}".format(task_name, memory_1, se_task_name, memory_2, t_r_1, t_r_2))
                    
                    if t_r_1 == 1e10 or t_r_2 == 1e10:
                        continue

                  
                    t_1 = infer_factor[0] * t_r_1
                    t_2 = infer_factor[1] * t_r_2
                    if 1/t_2 > throughput:
                        throughput = 1/t_2
                        selected_task = self.job_list[i]
                        selected_memory_1 = memory_1
                        selected_memory_2 = memory_2
                
        if selected_task in self.job_list:
            self.job_list.remove(selected_task)
        timestamp('[SCHEDULE]',f'{task_name},{selected_task[1] if selected_task is not None else None},{selected_memory_1}, {selected_memory_2}')
        return selected_task, self.job_list  , selected_memory_1, selected_memory_2          
class Schedule0NoInterference(ScheduleBase):
    def __init__(self, job_list, model_path):
        """ """
        super(Schedule0NoInterference, self).__init__(job_list)
        self.job_list = job_list
        self.predict_model = InterferencePredict(model_path)
        self.recomputation = RecomputationPredict() 
    @get_first_job_wrapper
    def get_first_job(self):
        min_training_time = 1e9
        for i in range(len(self.job_list)):
            task_name = self.job_list[i][1]
            task_info = TaskInfo(task_name)
            training_time = task_info.get_training_time()
            if training_time < min_training_time:
                min_training_time = training_time
                se_job = self.job_list[i]
        
        self.job_list.remove(se_job)
        return se_job, self.job_list
            
               
    @get_co_job_wrapper
    def get_co_job(self, job):
        
        task_name = job[1]
        task_info = TaskInfo(task_name)
        peak_memory = task_info.get_peak_memory()
        metrics = task_info.get_metrics()
        training_time = task_info.get_training_time()
        throughput = 0

        if len(self.job_list) == 0:
            return None, self.job_list, peak_memory, 0
        selected_task = None
        selected_memory_1 = peak_memory
        selected_memory_2 = 0
        for i in range(len(self.job_list)):
            se_task_name = self.job_list[i][1]
            se_task_info = TaskInfo(se_task_name)
            se_peak_memory = se_task_info.get_peak_memory()
            se_metrics = se_task_info.get_metrics()
            se_training_time = se_task_info.get_training_time()
          
            data = [*metrics, *se_metrics]
            infer_factor = self.predict_model.predict_interference(metrics, se_metrics)
            infer_factor = (1,1)
            timestamp('[INF_FACT]', str((task_name, se_task_name, infer_factor)))
          
            if se_peak_memory + peak_memory < DEVICE_MEMORY:
                #t_1 = infer_factor[0] * training_time # todo 需要敏感度系数吗？看结果误差 wsq
                t_2 = infer_factor[1] * se_training_time
                if 1/t_2 > throughput:
                    throughput = 1/t_2
                    selected_task = self.job_list[i]
                    selected_memory_1 = peak_memory
                    selected_memory_2 = se_peak_memory
            else:
                compressed_memory = se_peak_memory + peak_memory - DEVICE_MEMORY
                for j in range(-1, ceil(compressed_memory)*2+2):
                    memory_1 = floor(peak_memory * 2 - j ) / 2
                    memory_2 = floor(se_peak_memory * 2 - (ceil(compressed_memory)*2 - j))/2

                    timestamp("[VERBOSE_SCHEDULE]", "Trying {}-{} {}-{}".format(task_name, memory_1, se_task_name, memory_2))

                    t_r_1 = self.recomputation.predict_recomputation(memory_1, task_name, training_time)
                    t_r_2 = self.recomputation.predict_recomputation(memory_2, se_task_name, se_training_time)
                    
                    timestamp("[VERBOSE_SCHEDULE]", "{}-{} {}-{} {} {}".format(task_name, memory_1, se_task_name, memory_2, t_r_1, t_r_2))
                    
                    if t_r_1 == 1e10 or t_r_2 == 1e10:
                        continue

                  
                    t_1 = infer_factor[0] * t_r_1
                    t_2 = infer_factor[1] * t_r_2
                    if 1/t_2 > throughput:
                        throughput = 1/t_2
                        selected_task = self.job_list[i]
                        selected_memory_1 = memory_1
                        selected_memory_2 = memory_2
                
        if selected_task in self.job_list:
            self.job_list.remove(selected_task)
        timestamp('[SCHEDULE]',f'{task_name},{selected_task[1] if selected_task is not None else None},{selected_memory_1}, {selected_memory_2}')
        return selected_task, self.job_list  , selected_memory_1, selected_memory_2          
    
                
        
class Schedule1NoInterference(ScheduleBase):
    def __init__(self, job_list, model_path):
        """ """
        super(Schedule1NoInterference, self).__init__(job_list)
        self.job_list = job_list
        self.predict_model = InterferencePredict(model_path)
        self.recomputation = RecomputationPredict() 
    
    @get_first_job_wrapper
    def get_first_job(self):
        min_training_time = 1e9
        for i in range(len(self.job_list)):
            task_name = self.job_list[i][1]
            task_info = TaskInfo(task_name)
            training_time = task_info.get_training_time()
            if training_time < min_training_time:
                min_training_time = training_time
                se_job = self.job_list[i]
        
        self.job_list.remove(se_job)
        return se_job, self.job_list
            
               
    @get_co_job_wrapper
    def get_co_job(self, job):
        
        task_name = job[1]
        task_info = TaskInfo(task_name)
        peak_memory = task_info.get_peak_memory()
        metrics = task_info.get_metrics()
        training_time = task_info.get_training_time()
        throughput = 0

        if len(self.job_list) == 0:
            return None, self.job_list, peak_memory, 0
        selected_task = None
        selected_memory_1 = peak_memory
        selected_memory_2 = 0
        for i in range(len(self.job_list)):
            se_task_name = self.job_list[i][1]
            se_task_info = TaskInfo(se_task_name)
            se_peak_memory = se_task_info.get_peak_memory()
            se_metrics = se_task_info.get_metrics()
            se_training_time = se_task_info.get_training_time()
          
            data = [*metrics, *se_metrics]
            infer_factor = self.predict_model.predict_interference(metrics, se_metrics)
            infer_factor = [1,1]
            timestamp('[INF_FACT]', str((task_name, se_task_name, infer_factor)))
          
            if se_peak_memory + peak_memory < DEVICE_MEMORY:
                t_1 = infer_factor[0] * training_time # todo 需要敏感度系数吗？看结果误差 wsq
                t_2 = infer_factor[1] * se_training_time
                if 1/t_1 + 1/t_2 > throughput:
                    throughput = 1/t_1 + 1/t_2
                    selected_task = self.job_list[i]
                    selected_memory_1 = peak_memory
                    selected_memory_2 = se_peak_memory
            else:
                compressed_memory = se_peak_memory + peak_memory - DEVICE_MEMORY
                for j in range(-1, ceil(compressed_memory)*2+2):
                    memory_1 = floor(peak_memory * 2 - j ) / 2
                    memory_2 = floor(se_peak_memory * 2 - (ceil(compressed_memory)*2 - j))/2


                    timestamp("[VERBOSE_SCHEDULE]", "Trying {}-{} {}-{}".format(task_name, memory_1, se_task_name, memory_2))

                    t_r_1 = self.recomputation.predict_recomputation(memory_1, task_name, training_time)
                    t_r_2 = self.recomputation.predict_recomputation(memory_2, se_task_name, se_training_time)
                    
                    timestamp("[VERBOSE_SCHEDULE]", "{}-{} {}-{} {} {}".format(task_name, memory_1, se_task_name, memory_2, t_r_1, t_r_2))
                    if t_r_1 == 1e10 or t_r_2 == 1e10:
                        continue

                  
                    t_1 = infer_factor[0] * t_r_1
                    t_2 = infer_factor[1] * t_r_2
                    if 1/t_1 + 1/t_2 > throughput:
                        throughput = 1/t_1 + 1/t_2
                        selected_task = self.job_list[i]
                        selected_memory_1 = memory_1
                        selected_memory_2 = memory_2
                
        if selected_task in self.job_list:
            self.job_list.remove(selected_task)
        timestamp('[SCHEDULE]',f'{task_name},{selected_task[1] if selected_task is not None else None}, {selected_memory_1}, {selected_memory_2}')
        return selected_task, self.job_list  , selected_memory_1, selected_memory_2          
class Schedule1(ScheduleBase):
    def __init__(self, job_list, model_path):
        """ """
        super(Schedule1, self).__init__(job_list)
        self.job_list = job_list
        self.predict_model = InterferencePredict(model_path)
        self.recomputation = RecomputationPredict() 
    
    @get_first_job_wrapper
    def get_first_job(self):
        min_training_time = 1e9
        for i in range(len(self.job_list)):
            task_name = self.job_list[i][1]
            task_info = TaskInfo(task_name)
            training_time = task_info.get_training_time()
            if training_time < min_training_time:
                min_training_time = training_time
                se_job = self.job_list[i]
        
        self.job_list.remove(se_job)
        return se_job, self.job_list
            
               
    @get_co_job_wrapper
    def get_co_job(self, job):
        
        task_name = job[1]
        task_info = TaskInfo(task_name)
        peak_memory = task_info.get_peak_memory()
        metrics = task_info.get_metrics()
        training_time = task_info.get_training_time()
        throughput = 0

        if len(self.job_list) == 0:
            return None, self.job_list, peak_memory, 0
        selected_task = None
        selected_memory_1 = peak_memory
        selected_memory_2 = 0
        for i in range(len(self.job_list)):
            se_task_name = self.job_list[i][1]
            se_task_info = TaskInfo(se_task_name)
            se_peak_memory = se_task_info.get_peak_memory()
            se_metrics = se_task_info.get_metrics()
            se_training_time = se_task_info.get_training_time()
          
            data = [*metrics, *se_metrics]
            infer_factor = self.predict_model.predict_interference(metrics, se_metrics)
            timestamp('[INF_FACT]', str((task_name, se_task_name, infer_factor)))
          
            if se_peak_memory + peak_memory < DEVICE_MEMORY:
                t_1 = infer_factor[0] * training_time # todo 需要敏感度系数吗？看结果误差 wsq
                t_2 = infer_factor[1] * se_training_time
                if 1/t_1 + 1/t_2 > throughput:
                    throughput = 1/t_1 + 1/t_2
                    selected_task = self.job_list[i]
                    selected_memory_1 = peak_memory
                    selected_memory_2 = se_peak_memory
            else:
                compressed_memory = se_peak_memory + peak_memory - DEVICE_MEMORY
                for j in range(-1, ceil(compressed_memory)*2+2):
                    memory_1 = floor(peak_memory * 2 - j ) / 2
                    memory_2 = floor(se_peak_memory * 2 - (ceil(compressed_memory)*2 - j))/2


                    timestamp("[VERBOSE_SCHEDULE]", "Trying {}-{} {}-{}".format(task_name, memory_1, se_task_name, memory_2))

                    t_r_1 = self.recomputation.predict_recomputation(memory_1, task_name, training_time)
                    t_r_2 = self.recomputation.predict_recomputation(memory_2, se_task_name, se_training_time)
                    
                    timestamp("[VERBOSE_SCHEDULE]", "{}-{} {}-{} {} {}".format(task_name, memory_1, se_task_name, memory_2, t_r_1, t_r_2))
                    if t_r_1 == 1e10 or t_r_2 == 1e10:
                        continue

                  
                    t_1 = infer_factor[0] * t_r_1
                    t_2 = infer_factor[1] * t_r_2
                    if 1/t_1 + 1/t_2 > throughput:
                        throughput = 1/t_1 + 1/t_2
                        selected_task = self.job_list[i]
                        selected_memory_1 = memory_1
                        selected_memory_2 = memory_2
                
        if selected_task in self.job_list:
            self.job_list.remove(selected_task)
        timestamp('[SCHEDULE]',f'{task_name},{selected_task[1] if selected_task is not None else None}, {selected_memory_1}, {selected_memory_2}')
        return selected_task, self.job_list  , selected_memory_1, selected_memory_2          
    
                
        
class Schedule2(ScheduleBase):
    def __init__(self, job_list, model_path):
        """ """
        super(Schedule2, self).__init__(job_list)
        self.job_list = job_list
        self.predict_model = InterferencePredict(model_path)
        self.recomputation = RecomputationPredict() 

    @time
    @get_first_job_wrapper
    def get_first_job(self):
        se_job = max(
            self.job_list, 
            key=lambda job: TaskInfo(job[1]).get_peak_memory())
        #min_memory_threshold = 1e9
        #for i in range(len(self.job_list)):
        #    task_name = self.job_list[i][1]
        #    task_info = TaskInfo(task_name)
        #    memory_threshold = task_info.get_memory_threshold()
        #    if memory_threshold < min_memory_threshold:
        #        min_memory_threshold = memory_threshold
        #        se_job = self.job_list[i]
        
        self.job_list.remove(se_job)
        return se_job, self.job_list
            
    @time         
    @get_co_job_wrapper
    def get_co_job(self, job):
        
        task_name = job[1]
        task_info = TaskInfo(task_name)
        peak_memory = task_info.get_peak_memory()
        metrics = task_info.get_metrics()
        training_time = task_info.get_training_time()
        throughput = 0

        if len(self.job_list) == 0:
            return None, self.job_list, peak_memory, 0
        selected_task = None
        selected_memory_1 = peak_memory
        selected_memory_2 = 0
        job_groups = {}
        level_boundary = [10, 20, 32]
        for se_task in self.job_list:
            se_task_name = se_task[1]
            se_task_info = TaskInfo(se_task_name)
            se_peak_memory = se_task_info.get_peak_memory()
            se_metrics = se_task_info.get_metrics()
            se_training_time = se_task_info.get_training_time()
            se_memory_threshold = se_task_info.get_memory_threshold()
            for i, boundary in enumerate(level_boundary):
                if se_memory_threshold < boundary:
                    se_group = job_groups.get(i, [])
                    se_group.append(se_task)
                    job_groups[i] = se_group
                    break
            
            
        for ii in reversed(range(len(level_boundary))):
            job_group = job_groups.get(ii, [])
            for i in range(len(job_group)):
                se_task_name = job_group[i][1]
                se_task_info = TaskInfo(se_task_name)
                se_peak_memory = se_task_info.get_peak_memory()
                se_metrics = se_task_info.get_metrics()
                se_training_time = se_task_info.get_training_time()

                data = [*metrics, *se_metrics]
                infer_factor = self.predict_model.predict_interference(metrics, se_metrics)
                timestamp('[INF_FACT]', str((task_name, se_task_name, infer_factor)))

                if se_peak_memory + peak_memory < DEVICE_MEMORY:
                    t_1 = infer_factor[0] * training_time # todo 需要敏感度系数吗？看结果误差 wsq
                    t_2 = infer_factor[1] * se_training_time
                    if 1/t_1 + 1/t_2 > throughput:
                        throughput = 1/t_1 + 1/t_2
                        selected_task = job_group[i]
                        selected_memory_1 = peak_memory
                        selected_memory_2 = se_peak_memory
                else:
                    compressed_memory = se_peak_memory + peak_memory - DEVICE_MEMORY
                    for j in range(-1, ceil(compressed_memory)*2+2):
                        memory_1 = floor(peak_memory * 2 - j ) / 2
                        memory_2 = floor(se_peak_memory * 2 - (ceil(compressed_memory)*2 - j))/2


                        timestamp("[VERBOSE_SCHEDULE]", "Trying {}-{} {}-{}".format(task_name, memory_1, se_task_name, memory_2))

                        t_r_1 = self.recomputation.predict_recomputation(memory_1, task_name, training_time)
                        t_r_2 = self.recomputation.predict_recomputation(memory_2, se_task_name, se_training_time)
                        
                        timestamp("[VERBOSE_SCHEDULE]", "{}-{} {}-{} {} {}".format(task_name, memory_1, se_task_name, memory_2, t_r_1, t_r_2))
                        if t_r_1 == 1e10 or t_r_2 == 1e10:
                            continue

                        
                        t_1 = infer_factor[0] * t_r_1
                        t_2 = infer_factor[1] * t_r_2
                        if 1/t_1 + 1/t_2 > throughput:
                            throughput = 1/t_1 + 1/t_2
                            selected_task = job_group[i]
                            selected_memory_1 = memory_1
                            selected_memory_2 = memory_2
                
            if selected_task is not None:
                break
        if selected_task in self.job_list:
            self.job_list.remove(selected_task)
        
        timestamp('[SCHEDULE]',f'{task_name},{selected_task[1] if selected_task is not None else None}, {selected_memory_1}, {selected_memory_2}')
        return selected_task, self.job_list  , selected_memory_1, selected_memory_2          
class Schedule3(ScheduleBase):
    def __init__(self, job_list, model_path):
        """ """
        super(Schedule3, self).__init__(job_list)
        self.job_list = job_list
        self.predict_model = InterferencePredict(model_path)
        self.recomputation = RecomputationPredict() 

    @time
    @get_first_job_wrapper
    def get_first_job(self):
        se_job = max(
            self.job_list, 
            key=lambda job: TaskInfo(job[1]).get_peak_memory())
        #min_memory_threshold = 1e9
        #for i in range(len(self.job_list)):
        #    task_name = self.job_list[i][1]
        #    task_info = TaskInfo(task_name)
        #    memory_threshold = task_info.get_memory_threshold()
        #    if memory_threshold < min_memory_threshold:
        #        min_memory_threshold = memory_threshold
        #        se_job = self.job_list[i]
        
        self.job_list.remove(se_job)
        return se_job, self.job_list
            
    @time         
    @get_co_job_wrapper
    def get_co_job(self, job):
        
        task_name = job[1]
        task_info = TaskInfo(task_name)
        peak_memory = task_info.get_peak_memory()
        metrics = task_info.get_metrics()
        training_time = task_info.get_training_time()
        throughput = 0

        if len(self.job_list) == 0:
            return None, self.job_list, peak_memory, 0
        selected_task = None
        selected_memory_1 = peak_memory
        selected_memory_2 = 0
        job_groups = {}
        level_boundary = [10, 20, 32]
        for se_task in self.job_list:
            se_task_name = se_task[1]
            se_task_info = TaskInfo(se_task_name)
            se_peak_memory = se_task_info.get_peak_memory()
            se_metrics = se_task_info.get_metrics()
            se_training_time = se_task_info.get_training_time()
            se_memory_threshold = se_task_info.get_memory_threshold()
            for i, boundary in enumerate(level_boundary):
                if se_memory_threshold < boundary:
                    se_group = job_groups.get(i, [])
                    se_group.append(se_task)
                    job_groups[i] = se_group
                    break
            
            
        for ii in reversed(range(len(level_boundary))):
            job_group = job_groups.get(ii, [])
            for i in range(len(job_group)):
                se_task_name = job_group[i][1]
                se_task_info = TaskInfo(se_task_name)
                se_peak_memory = se_task_info.get_peak_memory()
                se_metrics = se_task_info.get_metrics()
                se_training_time = se_task_info.get_training_time()

                data = [*metrics, *se_metrics]
                infer_factor = self.predict_model.predict_interference(metrics, se_metrics)
                timestamp('[INF_FACT]', str((task_name, se_task_name, infer_factor)))

                if se_peak_memory + peak_memory < DEVICE_MEMORY:
                    t_1 = infer_factor[0] * training_time # todo 需要敏感度系数吗？看结果误差 wsq
                    t_2 = infer_factor[1] * se_training_time
                    
                    new_time = 1
                    # speedup = (1/t_1 + 1/t_2) / ((1/training_time + 1/se_training_time)/2)
                    speedup = (2*training_time + se_training_time) / (t_1+t_2)
                    
                    weight = speedup

                    timestamp('[WEIGHT no compress]', f'{task_name} {se_task_name} {weight}')
                    
                    if weight > throughput:
                        throughput = weight
                        selected_task = job_group[i]
                        selected_memory_1 = peak_memory
                        selected_memory_2 = se_peak_memory
                else:
                    compressed_memory = se_peak_memory + peak_memory - DEVICE_MEMORY
                    for j in range(-1, ceil(compressed_memory)*2+2):
                        memory_1 = floor(peak_memory * 2 - j ) / 2
                        memory_2 = floor(se_peak_memory * 2 - (ceil(compressed_memory)*2 - j))/2

                        timestamp("[VERBOSE_SCHEDULE]", "Trying {}-{} {}-{}".format(task_name, memory_1, se_task_name, memory_2))

                        t_r_1 = self.recomputation.predict_recomputation(memory_1, task_name, training_time)
                        t_r_2 = self.recomputation.predict_recomputation(memory_2, se_task_name, se_training_time)
                        
                        if t_r_1 == 1e10 or t_r_2 == 1e10:
                            continue
                        
                        t_1 = infer_factor[0] * t_r_1
                        t_2 = infer_factor[1] * t_r_2
                        
                        # weight = ((1/t_1 + 1/t_2) / (1/training_time + 1/se_training_time) / 2)
                        weight = (2*training_time + se_training_time) / (t_1+t_2)
                        
                        timestamp("[WEIGHT]", f"{task_name} {se_task_name} {weight}")
                        if weight > throughput:
                            throughput = weight
                            selected_task = job_group[i]
                            selected_memory_1 = memory_1
                            selected_memory_2 = memory_2
                
            if selected_task is not None:
                break
        if selected_task in self.job_list:
            self.job_list.remove(selected_task)
        
        timestamp('[SCHEDULE]',f'{task_name},{selected_task[1] if selected_task is not None else None}, {selected_memory_1}, {selected_memory_2}')
        return selected_task, self.job_list  , selected_memory_1, selected_memory_2          

class ScheduleLucid(ScheduleBase):
    def __init__(self, job_list, prediction_path='./lucid_prediction1.json'):
        """ """
        super(ScheduleLucid, self).__init__(job_list)
        self.job_list = job_list
        self.lucid_prediction_data = {}
        with open(prediction_path) as f:
            self.lucid_prediction_data = json.load(f)
    
    @get_first_job_wrapper
    def get_first_job(self):
        min_training_time = 1e9
        for i in range(len(self.job_list)):
            task_name = self.job_list[i][1]
            task_info = TaskInfo(task_name)
            training_time = task_info.get_training_time()
            if training_time < min_training_time:
                min_training_time = training_time
                se_job = self.job_list[i]
        
        self.job_list.remove(se_job)
        return se_job, self.job_list
            
               
    @get_co_job_wrapper
    def get_co_job(self, job):
        task_name = job[1]
        task_info = TaskInfo(task_name)
        peak_memory = task_info.get_peak_memory()
        metrics = task_info.get_metrics()
        training_time = task_info.get_training_time()

        gss_threshold = 2

        if len(self.job_list) == 0:
            return None, self.job_list, peak_memory, 0
        selected_task = None
        selected_memory_1 = peak_memory
        selected_memory_2 = 0
        for i in range(len(self.job_list)):
            se_task_name = self.job_list[i][1]
            se_task_info = TaskInfo(se_task_name)
            se_peak_memory = se_task_info.get_peak_memory()
            se_metrics = se_task_info.get_metrics()
            se_training_time = se_task_info.get_training_time()

            if se_peak_memory + peak_memory >= DEVICE_MEMORY:
                continue
          
            lucid_label_0 = self.lucid_prediction_data.get(task_name, 2.0)
            lucid_label_1 = self.lucid_prediction_data.get(se_task_name, 2.0)
            gpu_share_score = lucid_label_0 + lucid_label_1

            if gpu_share_score <= gss_threshold and training_time <= se_training_time * 2:
                selected_task = self.job_list[i]
                selected_memory_1 = peak_memory
                selected_memory_2 = se_peak_memory
                break
                
        if selected_task in self.job_list:
            self.job_list.remove(selected_task)
        timestamp('[SCHEDULE]',f'{task_name},{selected_task[1] if selected_task is not None else None}, {selected_memory_1}, {selected_memory_2}')
        return selected_task, self.job_list, selected_memory_1, selected_memory_2


class ScheduleRecompute0():
    def __init__(self, job_list, model_path):
        """ """
        self.job_list = job_list
        self.predict_model = InterferencePredict(model_path)
        self.recomputation = RecomputationPredict()
        self.sched_job_cnt = 0
    
    def get_first_job(self):
        se_job = self.job_list[0]
        self.job_list.remove(se_job)
        self.sched_job_cnt += 1
        
        return se_job, self.job_list

    def get_co_job(self, job):
        task_name = job[1]
        task_info = TaskInfo(task_name)
        peak_memory = task_info.get_peak_memory()
        metrics = task_info.get_metrics()
        training_time = task_info.get_training_time()
        
        selected_task = None
        selected_memory_1 = peak_memory
        selected_memory_2 = 0
        recomp_cost = 1e10
        
        if self.sched_job_cnt % 2 == 0:
            timestamp("[schedule_recompute]", "waiting for other parts in pair to be complete so return nothing.")
            self.in_pair = False
            return selected_task, self.job_list, selected_memory_1, selected_memory_2

        if len(self.job_list) == 0:
            return None, self.job_list, peak_memory, 0
        
        se_task_name = self.job_list[0][1]
        se_task_info = TaskInfo(se_task_name)
        se_peak_memory = se_task_info.get_peak_memory()
        se_metrics = se_task_info.get_metrics()
        se_training_time = se_task_info.get_training_time()
        
        if se_peak_memory + peak_memory < DEVICE_MEMORY:
            selected_task = self.job_list[0]
            selected_memory_1 = peak_memory
            selected_memory_2 = se_peak_memory
            print("[VERBOSE_SCHEDULE] No Compression: {}-{} {}-{}".format(task_name, selected_memory_1, se_task_name, selected_memory_2))
        else:
            compressed_memory = se_peak_memory + peak_memory - DEVICE_MEMORY
            for j in range(-1, ceil(compressed_memory)*2+2):
                memory_1 = floor(peak_memory * 2 - j ) / 2
                memory_2 = floor(se_peak_memory * 2 - (ceil(compressed_memory)*2 - j))/2

                timestamp("[VERBOSE_SCHEDULE]", "Trying {}-{} {}-{}".format(task_name, memory_1, se_task_name, memory_2))

                t_r_1 = self.recomputation.predict_recomputation(memory_1, task_name, training_time)
                t_r_2 = self.recomputation.predict_recomputation(memory_2, se_task_name, se_training_time)
                
                timestamp("[VERBOSE_SCHEDULE]", "{}-{} {}-{} {} {}".format(task_name, memory_1, se_task_name, memory_2, t_r_1, t_r_2))
                if t_r_1 == 1e10 or t_r_2 == 1e10:
                    continue
                
                total_recomp = t_r_1 + t_r_2
                if total_recomp < recomp_cost:
                    recomp_cost = total_recomp
                    selected_task = self.job_list[0]
                    selected_memory_1 = memory_1
                    selected_memory_2 = memory_2
                
        if selected_task in self.job_list:
            self.job_list.remove(selected_task)
            self.sched_job_cnt += 1
        timestamp('[SCHEDULE]',f'{task_name},{selected_task[1] if selected_task is not None else None}, {selected_memory_1}, {selected_memory_2}')
        return selected_task, self.job_list, selected_memory_1, selected_memory_2
    
class ScheduleRecompute1():
    def __init__(self, job_list, model_path):
        """ """
        self.job_list = job_list
        self.predict_model = InterferencePredict(model_path)
        self.recomputation = RecomputationPredict()
        self.sched_job_cnt = 0
    
    def get_first_job(self):
        se_job = self.job_list[0]
        self.job_list.remove(se_job)
        self.sched_job_cnt += 1
        
        return se_job, self.job_list

    def get_co_job(self, job):
        task_name = job[1]
        task_info = TaskInfo(task_name)
        peak_memory = task_info.get_peak_memory()
        metrics = task_info.get_metrics()
        training_time = task_info.get_training_time()
        
        selected_task = None
        selected_memory_1 = peak_memory
        selected_memory_2 = 0
        
        if self.sched_job_cnt % 2 == 0:
            timestamp("[schedule_recompute]", "waiting for other parts in pair to be complete so return nothing.")
            self.in_pair = False
            return selected_task, self.job_list, selected_memory_1, selected_memory_2

        if len(self.job_list) == 0:
            return None, self.job_list, peak_memory, 0
        
        recomp_cost = 3e10
        for i in range(len(self.job_list)):
            se_task_name = self.job_list[i][1]
            se_task_info = TaskInfo(se_task_name)
            se_peak_memory = se_task_info.get_peak_memory()
            se_metrics = se_task_info.get_metrics()
            se_training_time = se_task_info.get_training_time()
            
            if se_peak_memory + peak_memory < DEVICE_MEMORY:
                selected_task = self.job_list[i]
                selected_memory_1 = peak_memory
                selected_memory_2 = se_peak_memory
                print("[VERBOSE_SCHEDULE] No Compression: {}-{} {}-{}".format(task_name, selected_memory_1, se_task_name, selected_memory_2))
                break
            else:
                compressed_memory = se_peak_memory + peak_memory - DEVICE_MEMORY
                for j in range(-1, ceil(compressed_memory)*2+2):
                    memory_1 = floor(peak_memory * 2 - j ) / 2
                    memory_2 = floor(se_peak_memory * 2 - (ceil(compressed_memory)*2 - j))/2

                    t_r_1 = self.recomputation.predict_recomputation(memory_1, task_name, training_time)
                    t_r_2 = self.recomputation.predict_recomputation(memory_2, se_task_name, se_training_time)
                    if t_r_1 == 1e10 or t_r_2 == 1e10:
                        continue
                    
                    print("[VERBOSE_SCHEDULE] {}-{} {}-{} {} {}".format(task_name, memory_1, se_task_name, memory_2, t_r_1, t_r_2))
                    
                    total_recomp = t_r_1 + t_r_2
                    if total_recomp < recomp_cost:
                        recomp_cost = total_recomp
                        selected_task = self.job_list[i]
                        selected_memory_1 = memory_1
                        selected_memory_2 = memory_2
                
        if selected_task in self.job_list:
            self.job_list.remove(selected_task)
            self.sched_job_cnt += 1
        timestamp('[SCHEDULE]',f'{task_name},{selected_task[1] if selected_task is not None else None},{selected_memory_1}, {selected_memory_2}')
        return selected_task, self.job_list, selected_memory_1, selected_memory_2


class ScheduleRecompute2():
    def __init__(self, job_list, model_path):
        """ """
        self.job_list = job_list
        self.predict_model = InterferencePredict(model_path)
        self.recomputation = RecomputationPredict()
        self.sched_job_cnt = 0
    
    def get_first_job(self):
        se_job = self.job_list[0]
        self.job_list.remove(se_job)
        self.sched_job_cnt += 1
        
        return se_job, self.job_list
    
    def get_co_job(self, job):
        
        task_name = job[1]
        task_info = TaskInfo(task_name)
        peak_memory = task_info.get_peak_memory()
        metrics = task_info.get_metrics()
        training_time = task_info.get_training_time()
        throughput = 1

        if len(self.job_list) == 0:
            return None, self.job_list, peak_memory, 0
        
        selected_task = None
        selected_memory_1 = peak_memory
        selected_memory_2 = 0
        
        if self.sched_job_cnt % 2 == 0:
            timestamp("[schedule_recompute]", "waiting for other parts in pair to be complete so return nothing.")
            return selected_task, self.job_list, selected_memory_1, selected_memory_2
        
        job_groups = {}
        level_boundary = [10, 20, 32]
        for se_task in self.job_list:
            se_task_name = se_task[1]
            se_task_info = TaskInfo(se_task_name)
            se_peak_memory = se_task_info.get_peak_memory()
            se_training_time = se_task_info.get_training_time()
            se_memory_threshold = se_task_info.get_memory_threshold()
            for i, boundary in enumerate(level_boundary):
                if se_memory_threshold < boundary:
                    se_group = job_groups.get(i, [])
                    se_group.append(se_task)
                    job_groups[i] = se_group
                    break
            
            
        for ii in reversed(range(len(level_boundary))):
            job_group = job_groups.get(ii, [])
            recomp_cost = 3e10
            for i in range(len(job_group)):
                se_task_name = job_group[i][1]
                se_task_info = TaskInfo(se_task_name)
                se_peak_memory = se_task_info.get_peak_memory()
                se_training_time = se_task_info.get_training_time()

                if se_peak_memory + peak_memory < DEVICE_MEMORY:
                    selected_task = job_group[i]
                    selected_memory_1 = peak_memory
                    selected_memory_2 = se_peak_memory
                    recomp_cost = 0
                    print("[VERBOSE_SCHEDULE] No Compression: {}-{} {}-{}".format(task_name, selected_memory_1, se_task_name, selected_memory_2))
                    break
                else:
                    compressed_memory = se_peak_memory + peak_memory - DEVICE_MEMORY
                    for j in range(-1, ceil(compressed_memory)*2+2):
                        memory_1 = floor(peak_memory * 2 - j ) / 2
                        memory_2 = floor(se_peak_memory * 2 - (ceil(compressed_memory)*2 - j))/2

                        t_r_1 = self.recomputation.predict_recomputation(memory_1, task_name, training_time)
                        t_r_2 = self.recomputation.predict_recomputation(memory_2, se_task_name, se_training_time)
                        if t_r_1 == 1e10 or t_r_2 == 1e10:
                            continue
                        total_recomp = t_r_1 + t_r_2
                        print("[VERBOSE_SCHEDULE] {}-{} {}-{} {} {}".format(task_name, memory_1, se_task_name, memory_2, t_r_1, t_r_2))
                        if total_recomp < recomp_cost:
                            total_recomp = recomp_cost
                            selected_task = job_group[i]
                            selected_memory_1 = memory_1
                            selected_memory_2 = memory_2
                
            if selected_task is not None:
                break
        if selected_task in self.job_list:
            self.job_list.remove(selected_task)
            self.sched_job_cnt += 1
        timestamp("[SELECT]", f"{task_name} {selected_task} {throughput}")
        timestamp("[SE_MEMORY]", f"{selected_memory_1}, {selected_memory_2}")

        timestamp('[SCHEDULE]',f'{task_name},{selected_task[1] if selected_task is not None else None}, {selected_memory_1}, {selected_memory_2}')
        
        return selected_task, self.job_list  , selected_memory_1, selected_memory_2        
class ScheduleSimple():
    def __init__(self, job_list, model_path):
        """ """
        self.job_list = job_list
        self.predict_model = InterferencePredict(model_path)
        self.recomputation = RecomputationPredict() 
    
    def get_first_job(self):
        min_training_time = 1e9
        se_job = self.job_list[0] 
        self.job_list.remove(se_job)
        return se_job, self.job_list
            
               
    def get_co_job(self, job):
        if len(self.job_list) != 0:
            se_job, job_list = self.get_first_job()
            return se_job, self.job_list, float(job[2]), float(se_job[2])
        else:
            return None, self.job_list, 31, 0

class ScheduleRecompute0_static(ScheduleBase):
    def __init__(self, job_list, model_path):
        """ """
        super(ScheduleRecompute0_static, self).__init__(job_list)
        self.job_list = job_list
        self.predict_model = InterferencePredict(model_path)
        self.recomputation = RecomputationPredict()
        self.sched_job_cnt = 0
    
    @get_first_job_wrapper
    def get_first_job(self):
        se_job = self.job_list[0]
        self.job_list.remove(se_job)
        self.sched_job_cnt += 1
        
        return se_job, self.job_list

    @time         
    @get_co_job_wrapper
    def get_co_job(self, job):
        is_already_running = (len(job) == 5)
        if is_already_running:
            return self.get_co_job_running(job, job[4])
        else:
            return self.get_co_job_fresh(job)

    def get_co_job_fresh(self, job):
        task_name = job[1]
        task_info = TaskInfo(task_name)
        peak_memory = task_info.get_peak_memory()
        metrics = task_info.get_metrics()
        training_time = task_info.get_training_time()
        
        selected_task = None
        selected_memory_1 = peak_memory
        selected_memory_2 = 0
        recomp_cost = 1e10

        if len(self.job_list) == 0:
            return None, self.job_list, peak_memory, 0
        
        se_task_name = self.job_list[0][1]
        se_task_info = TaskInfo(se_task_name)
        se_peak_memory = se_task_info.get_peak_memory()
        se_metrics = se_task_info.get_metrics()
        se_training_time = se_task_info.get_training_time()
        
        if se_peak_memory + peak_memory < DEVICE_MEMORY:
            selected_task = self.job_list[0]
            selected_memory_1 = peak_memory
            selected_memory_2 = se_peak_memory
            timestamp("[VERBOSE_SCHEDULE]", "No Compression: {}-{} {}-{}".format(task_name, selected_memory_1, se_task_name, selected_memory_2))
        else:
            compressed_memory = se_peak_memory + peak_memory - DEVICE_MEMORY
            for j in range(-1, ceil(compressed_memory)*2+2):
                memory_1 = floor(peak_memory * 2 - j ) / 2
                memory_2 = floor(se_peak_memory * 2 - (ceil(compressed_memory)*2 - j))/2

                timestamp("[VERBOSE_SCHEDULE]", "Trying {}-{} {}-{}".format(task_name, memory_1, se_task_name, memory_2))

                t_r_1 = self.recomputation.predict_recomputation(memory_1, task_name, training_time)
                t_r_2 = self.recomputation.predict_recomputation(memory_2, se_task_name, se_training_time)
                
                timestamp("[VERBOSE_SCHEDULE]", "{}-{} {}-{} {} {}".format(task_name, memory_1, se_task_name, memory_2, t_r_1, t_r_2))
                if t_r_1 == 1e10 or t_r_2 == 1e10:
                    continue
                
                total_recomp = t_r_1 + t_r_2
                if total_recomp < recomp_cost:
                    recomp_cost = total_recomp
                    selected_task = self.job_list[0]
                    selected_memory_1 = memory_1
                    selected_memory_2 = memory_2
                    break
                
        if selected_task in self.job_list:
            self.job_list.remove(selected_task)
            self.sched_job_cnt += 1
        timestamp('[SCHEDULE]',f'{task_name},{selected_task[1] if selected_task is not None else None}, {selected_memory_1}, {selected_memory_2}')
        return selected_task, self.job_list, selected_memory_1, selected_memory_2

    def get_co_job_running(self, job, original_mem):
        task_name = job[1]
        task_info = TaskInfo(task_name)
        metrics = task_info.get_metrics()
        training_time = task_info.get_training_time()
        
        selected_task = None
        selected_memory_1 = original_mem
        selected_memory_2 = 0
        recomp_cost = 1e10

        if len(self.job_list) == 0:
            return None, self.job_list, original_mem, 0
        
        se_task_name = self.job_list[0][1]
        se_task_info = TaskInfo(se_task_name)
        se_peak_memory = se_task_info.get_peak_memory()
        se_metrics = se_task_info.get_metrics()
        se_training_time = se_task_info.get_training_time()
        
        if se_peak_memory + original_mem < DEVICE_MEMORY:
            selected_task = self.job_list[0]
            selected_memory_1 = original_mem
            selected_memory_2 = se_peak_memory
            timestamp("[VERBOSE_SCHEDULE]", "No Compression: {}-{} {}-{}".format(task_name, selected_memory_1, se_task_name, selected_memory_2))
        else:
            compressed_memory = se_peak_memory + original_mem - DEVICE_MEMORY
            memory_1 = original_mem
            memory_2 = floor((DEVICE_MEMORY - memory_1) * 2)/2

            timestamp("[VERBOSE_SCHEDULE]", "Trying {}-{} {}-{}".format(task_name, memory_1, se_task_name, memory_2))

            t_r_1 = self.recomputation.predict_recomputation(memory_1, task_name, training_time)
            t_r_2 = self.recomputation.predict_recomputation(memory_2, se_task_name, se_training_time)

            if t_r_1 == 1e10 or t_r_2 == 1e10:
                timestamp('[SCHEDULE]',f'{task_name},{selected_task[1] if selected_task is not None else None}, {selected_memory_1}, {selected_memory_2}')
                return None, self.job_list, original_mem, 0
            
            timestamp("[VERBOSE_SCHEDULE]", "{}-{} {}-{} {} {}".format(task_name, memory_1, se_task_name, memory_2, t_r_1, t_r_2))
            
            total_recomp = t_r_1 + t_r_2
            if total_recomp < recomp_cost:
                recomp_cost = total_recomp
                selected_task = self.job_list[0]
                selected_memory_1 = memory_1
                selected_memory_2 = memory_2
                
        if selected_task in self.job_list:
            self.job_list.remove(selected_task)
            self.sched_job_cnt += 1
        timestamp('[SCHEDULE]',f'{task_name},{selected_task[1] if selected_task is not None else None}, {selected_memory_1}, {selected_memory_2}')
        return selected_task, self.job_list, selected_memory_1, selected_memory_2

class ScheduleRecompute0_nocost(ScheduleBase):
    def __init__(self, job_list, model_path):
        """ """
        super(ScheduleRecompute0_nocost, self).__init__(job_list)
        self.job_list = job_list
        self.predict_model = InterferencePredict(model_path)
        self.recomputation = RecomputationPredict()
        self.sched_job_cnt = 0
    
    @get_first_job_wrapper
    def get_first_job(self):
        se_job = self.job_list[0]
        self.job_list.remove(se_job)
        self.sched_job_cnt += 1
        
        return se_job, self.job_list

    @get_co_job_wrapper
    def get_co_job(self, job):
        task_name = job[1]
        task_info = TaskInfo(task_name)
        peak_memory = task_info.get_peak_memory()
        metrics = task_info.get_metrics()
        training_time = task_info.get_training_time()
        
        selected_task = None
        selected_memory_1 = peak_memory
        selected_memory_2 = 0
        recomp_cost = 1e10

        if len(self.job_list) == 0:
            return None, self.job_list, peak_memory, 0
        
        se_task_name = self.job_list[0][1]
        se_task_info = TaskInfo(se_task_name)
        se_peak_memory = se_task_info.get_peak_memory()
        se_metrics = se_task_info.get_metrics()
        se_training_time = se_task_info.get_training_time()
        
        if se_peak_memory + peak_memory < DEVICE_MEMORY:
            selected_task = self.job_list[0]
            selected_memory_1 = peak_memory
            selected_memory_2 = se_peak_memory
            timestamp("[VERBOSE_SCHEDULE]", "No Compression: {}-{} {}-{}".format(task_name, selected_memory_1, se_task_name, selected_memory_2))
        else:
            compressed_memory = se_peak_memory + peak_memory - DEVICE_MEMORY
            for j in range(-1, ceil(compressed_memory)*2+2):
                memory_1 = floor(peak_memory * 2 - j ) / 2
                memory_2 = floor(se_peak_memory * 2 - (ceil(compressed_memory)*2 - j))/2
                #memory_2 = floor(se_peak_memory * 2 - j ) / 2
                #memory_1 = floor(peak_memory * 2 - (ceil(compressed_memory)*2 - j))/2

                timestamp("[VERBOSE_SCHEDULE]", "Trying {}-{} {}-{}".format(task_name, memory_1, se_task_name, memory_2))

                t_r_1 = self.recomputation.predict_recomputation(memory_1, task_name, training_time)
                t_r_2 = self.recomputation.predict_recomputation(memory_2, se_task_name, se_training_time)
                
                timestamp("[VERBOSE_SCHEDULE]", "{}-{} {}-{} {} {}".format(task_name, memory_1, se_task_name, memory_2, t_r_1, t_r_2))
                if t_r_1 == 1e10 or t_r_2 == 1e10:
                    continue
                
                total_recomp = t_r_1 + t_r_2
                if total_recomp < recomp_cost:
                    recomp_cost = total_recomp
                    selected_task = self.job_list[0]
                    selected_memory_1 = memory_1
                    selected_memory_2 = memory_2
                    break
                
        if selected_task in self.job_list:
            self.job_list.remove(selected_task)
            self.sched_job_cnt += 1
        timestamp('[SCHEDULE]',f'{task_name},{selected_task[1] if selected_task is not None else None}, {selected_memory_1}, {selected_memory_2}')
        return selected_task, self.job_list, selected_memory_1, selected_memory_2
    

class ScheduleRecompute0_pair_nocost(ScheduleBase):
    def __init__(self, job_list, model_path):
        """ """
        super(ScheduleRecompute0_pair_nocost, self).__init__(job_list)
        self.job_list = job_list
        self.predict_model = InterferencePredict(model_path)
        self.recomputation = RecomputationPredict()
        self.sched_job_cnt = 0
    
    @get_first_job_wrapper
    def get_first_job(self):
        se_job = self.job_list[0]
        self.job_list.remove(se_job)
        self.sched_job_cnt += 1
        
        return se_job, self.job_list

    @get_co_job_wrapper
    def get_co_job(self, job):
        task_name = job[1]
        task_info = TaskInfo(task_name)
        peak_memory = task_info.get_peak_memory()
        metrics = task_info.get_metrics()
        training_time = task_info.get_training_time()
        
        selected_task = None
        selected_memory_1 = peak_memory
        selected_memory_2 = 0
        recomp_cost = 1e10
        
        if self.sched_job_cnt % 2 == 0:
            timestamp("[schedule_recompute]", "waiting for other parts in pair to be complete so return nothing.")
            self.in_pair = False
            return selected_task, self.job_list, selected_memory_1, selected_memory_2

        if len(self.job_list) == 0:
            return None, self.job_list, peak_memory, 0
        
        se_task_name = self.job_list[0][1]
        se_task_info = TaskInfo(se_task_name)
        se_peak_memory = se_task_info.get_peak_memory()
        se_metrics = se_task_info.get_metrics()
        se_training_time = se_task_info.get_training_time()
        
        if se_peak_memory + peak_memory < DEVICE_MEMORY:
            selected_task = self.job_list[0]
            selected_memory_1 = peak_memory
            selected_memory_2 = se_peak_memory
            print("[VERBOSE_SCHEDULE] No Compression: {}-{} {}-{}".format(task_name, selected_memory_1, se_task_name, selected_memory_2))
        else:
            compressed_memory = se_peak_memory + peak_memory - DEVICE_MEMORY
            for j in range(-1, ceil(compressed_memory)*2+2):
                memory_2 = floor(se_peak_memory * 2 - j ) / 2
                memory_1 = floor(peak_memory * 2 - (ceil(compressed_memory)*2 - j))/2

                timestamp("[VERBOSE_SCHEDULE]", "Trying {}-{} {}-{}".format(task_name, memory_1, se_task_name, memory_2))

                t_r_1 = self.recomputation.predict_recomputation(memory_1, task_name, training_time)
                t_r_2 = self.recomputation.predict_recomputation(memory_2, se_task_name, se_training_time)
                
                timestamp("[VERBOSE_SCHEDULE]", "{}-{} {}-{} {} {}".format(task_name, memory_1, se_task_name, memory_2, t_r_1, t_r_2))
                if t_r_1 == 1e10 or t_r_2 == 1e10:
                    continue
                
                total_recomp = t_r_1 + t_r_2
                if total_recomp < recomp_cost:
                    recomp_cost = total_recomp
                    selected_task = self.job_list[0]
                    selected_memory_1 = memory_1
                    selected_memory_2 = memory_2
                    break
                
        if selected_task in self.job_list:
            self.job_list.remove(selected_task)
            self.sched_job_cnt += 1
        timestamp('[SCHEDULE]',f'{task_name},{selected_task[1] if selected_task is not None else None}, {selected_memory_1}, {selected_memory_2}')
        return selected_task, self.job_list, selected_memory_1, selected_memory_2

class Schedule2NoInterference(ScheduleBase):
    def __init__(self, job_list, model_path):
        """ """
        super(Schedule2NoInterference, self).__init__(job_list)
        self.job_list = job_list
        self.predict_model = InterferencePredict(model_path)
        self.recomputation = RecomputationPredict() 

    @time
    @get_first_job_wrapper
    def get_first_job(self):
        se_job = max(
            self.job_list, 
            key=lambda job: TaskInfo(job[1]).get_peak_memory())
        #min_memory_threshold = 1e9
        #for i in range(len(self.job_list)):
        #    task_name = self.job_list[i][1]
        #    task_info = TaskInfo(task_name)
        #    memory_threshold = task_info.get_memory_threshold()
        #    if memory_threshold < min_memory_threshold:
        #        min_memory_threshold = memory_threshold
        #        se_job = self.job_list[i]
        
        self.job_list.remove(se_job)
        return se_job, self.job_list
            
    @time         
    @get_co_job_wrapper
    def get_co_job(self, job):
        
        task_name = job[1]
        task_info = TaskInfo(task_name)
        peak_memory = task_info.get_peak_memory()
        metrics = task_info.get_metrics()
        training_time = task_info.get_training_time()
        throughput = 0

        if len(self.job_list) == 0:
            return None, self.job_list, peak_memory, 0
        selected_task = None
        selected_memory_1 = peak_memory
        selected_memory_2 = 0
        job_groups = {}
        level_boundary = [10, 20, 32]
        for se_task in self.job_list:
            se_task_name = se_task[1]
            se_task_info = TaskInfo(se_task_name)
            se_peak_memory = se_task_info.get_peak_memory()
            se_metrics = se_task_info.get_metrics()
            se_training_time = se_task_info.get_training_time()
            se_memory_threshold = se_task_info.get_memory_threshold()
            for i, boundary in enumerate(level_boundary):
                if se_memory_threshold < boundary:
                    se_group = job_groups.get(i, [])
                    se_group.append(se_task)
                    job_groups[i] = se_group
                    break
            
            
        for ii in reversed(range(len(level_boundary))):
            job_group = job_groups.get(ii, [])
            for i in range(len(job_group)):
                se_task_name = job_group[i][1]
                se_task_info = TaskInfo(se_task_name)
                se_peak_memory = se_task_info.get_peak_memory()
                se_metrics = se_task_info.get_metrics()
                se_training_time = se_task_info.get_training_time()

                data = [*metrics, *se_metrics]
                # infer_factor = self.predict_model.predict_interference(metrics, se_metrics)
                infer_factor = [1,1]
                timestamp('[INF_FACT]', str((task_name, se_task_name, infer_factor)))

                if se_peak_memory + peak_memory < DEVICE_MEMORY:
                    t_1 = infer_factor[0] * training_time # todo 需要敏感度系数吗？看结果误差 wsq
                    t_2 = infer_factor[1] * se_training_time
                    if 1/t_1 + 1/t_2 > throughput:
                        throughput = 1/t_1 + 1/t_2
                        selected_task = job_group[i]
                        selected_memory_1 = peak_memory
                        selected_memory_2 = se_peak_memory
                else:
                    compressed_memory = se_peak_memory + peak_memory - DEVICE_MEMORY
                    for j in range(-1, ceil(compressed_memory)*2+2):
                        memory_1 = floor(peak_memory * 2 - j ) / 2
                        memory_2 = floor(se_peak_memory * 2 - (ceil(compressed_memory)*2 - j))/2


                        timestamp("[VERBOSE_SCHEDULE]", "Trying {}-{} {}-{}".format(task_name, memory_1, se_task_name, memory_2))

                        t_r_1 = self.recomputation.predict_recomputation(memory_1, task_name, training_time)
                        t_r_2 = self.recomputation.predict_recomputation(memory_2, se_task_name, se_training_time)
                        
                        timestamp("[VERBOSE_SCHEDULE]", "{}-{} {}-{} {} {}".format(task_name, memory_1, se_task_name, memory_2, t_r_1, t_r_2))
                        if t_r_1 == 1e10 or t_r_2 == 1e10:
                            continue

                        
                        t_1 = infer_factor[0] * t_r_1
                        t_2 = infer_factor[1] * t_r_2
                        if 1/t_1 + 1/t_2 > throughput:
                            throughput = 1/t_1 + 1/t_2
                            selected_task = job_group[i]
                            selected_memory_1 = memory_1
                            selected_memory_2 = memory_2
                
            if selected_task is not None:
                break
        if selected_task in self.job_list:
            self.job_list.remove(selected_task)
        
        timestamp('[SCHEDULE]',f'{task_name},{selected_task[1] if selected_task is not None else None}, {selected_memory_1}, {selected_memory_2}')
        return selected_task, self.job_list  , selected_memory_1, selected_memory_2          


class Schedule2NoRecomputation(ScheduleBase):
    def __init__(self, job_list, model_path):
        """ """
        super(Schedule2NoRecomputation, self).__init__(job_list)
        self.job_list = job_list
        self.predict_model = InterferencePredict(model_path)
        self.recomputation = RecomputationPredict() 

    @time
    @get_first_job_wrapper
    def get_first_job(self):
        se_job = max(
            self.job_list, 
            key=lambda job: TaskInfo(job[1]).get_peak_memory())
        #min_memory_threshold = 1e9
        #for i in range(len(self.job_list)):
        #    task_name = self.job_list[i][1]
        #    task_info = TaskInfo(task_name)
        #    memory_threshold = task_info.get_memory_threshold()
        #    if memory_threshold < min_memory_threshold:
        #        min_memory_threshold = memory_threshold
        #        se_job = self.job_list[i]
        
        self.job_list.remove(se_job)
        return se_job, self.job_list
            
    @time         
    @get_co_job_wrapper
    def get_co_job(self, job):
        
        task_name = job[1]
        task_info = TaskInfo(task_name)
        peak_memory = task_info.get_peak_memory()
        metrics = task_info.get_metrics()
        training_time = task_info.get_training_time()
        throughput = 0

        if len(self.job_list) == 0:
            return None, self.job_list, peak_memory, 0
        selected_task = None
        selected_memory_1 = peak_memory
        selected_memory_2 = 0
        job_groups = {}
        level_boundary = [10, 20, 32]
        for se_task in self.job_list:
            se_task_name = se_task[1]
            se_task_info = TaskInfo(se_task_name)
            se_peak_memory = se_task_info.get_peak_memory()
            se_metrics = se_task_info.get_metrics()
            se_training_time = se_task_info.get_training_time()
            se_memory_threshold = se_task_info.get_memory_threshold()
            for i, boundary in enumerate(level_boundary):
                if se_memory_threshold < boundary:
                    se_group = job_groups.get(i, [])
                    se_group.append(se_task)
                    job_groups[i] = se_group
                    break
            
            
        for ii in reversed(range(len(level_boundary))):
            job_group = job_groups.get(ii, [])
            for i in range(len(job_group)):
                se_task_name = job_group[i][1]
                se_task_info = TaskInfo(se_task_name)
                se_peak_memory = se_task_info.get_peak_memory()
                se_metrics = se_task_info.get_metrics()
                se_training_time = se_task_info.get_training_time()

                data = [*metrics, *se_metrics]
                infer_factor = self.predict_model.predict_interference(metrics, se_metrics)
                timestamp('[INF_FACT]', str((task_name, se_task_name, infer_factor)))

                if se_peak_memory + peak_memory < DEVICE_MEMORY:
                    t_1 = infer_factor[0] * training_time # todo 需要敏感度系数吗？看结果误差 wsq
                    t_2 = infer_factor[1] * se_training_time
                    if 1/t_1 + 1/t_2 > throughput:
                        throughput = 1/t_1 + 1/t_2
                        selected_task = job_group[i]
                        selected_memory_1 = peak_memory
                        selected_memory_2 = se_peak_memory
                else:
                    compressed_memory = se_peak_memory + peak_memory - DEVICE_MEMORY
                    for j in range(-1, ceil(compressed_memory)*2+2):
                        memory_1 = floor(peak_memory * 2 - j ) / 2
                        memory_2 = floor(se_peak_memory * 2 - (ceil(compressed_memory)*2 - j))/2


                        timestamp("[VERBOSE_SCHEDULE]", "Trying {}-{} {}-{}".format(task_name, memory_1, se_task_name, memory_2))

                        t_r_1 = self.recomputation.predict_recomputation(memory_1, task_name, training_time)
                        t_r_2 = self.recomputation.predict_recomputation(memory_2, se_task_name, se_training_time)
                        
                        timestamp("[VERBOSE_SCHEDULE]", "{}-{} {}-{} {} {}".format(task_name, memory_1, se_task_name, memory_2, t_r_1, t_r_2))
                        if t_r_1 == 1e10 or t_r_2 == 1e10:
                            continue

                        
                        # t_1 = infer_factor[0] * t_r_1
                        # t_2 = infer_factor[1] * t_r_2
                        t_1 = infer_factor[0] * training_time
                        t_2 = infer_factor[1] * se_training_time
                        if 1/t_1 + 1/t_2 > throughput:
                            throughput = 1/t_1 + 1/t_2
                            selected_task = job_group[i]
                            selected_memory_1 = memory_1
                            selected_memory_2 = memory_2
                
            if selected_task is not None:
                break
        if selected_task in self.job_list:
            self.job_list.remove(selected_task)
        
        timestamp('[SCHEDULE]',f'{task_name},{selected_task[1] if selected_task is not None else None}, {selected_memory_1}, {selected_memory_2}')
        return selected_task, self.job_list  , selected_memory_1, selected_memory_2          
