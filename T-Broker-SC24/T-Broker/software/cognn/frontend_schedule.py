import threading
import json
import select
import torch
import importlib
import time
import os
import subprocess
from timeit import default_timer as timer

from util.util import timestamp
from cognn.policy import *
from cognn.schedule import Schedule0, Schedule1, Schedule2, Schedule3,ScheduleLucid, ScheduleRecompute0, ScheduleRecompute1, ScheduleRecompute2, ScheduleSimple, Schedule2NoRecomputation, get_schedule_time,ScheduleRecompute0_pair_nocost,ScheduleRecompute0_nocost,ScheduleRecompute0_static, Schedule2NoInterference,Schedule0NoInterference,Schedule1NoInterference

from cognn.task_info import TaskInfo

class FrontendScheduleThd(threading.Thread):
    def __init__(self, model_list, qin, worker_list, policy):
        super(FrontendScheduleThd, self).__init__()
        self.model_list = model_list
        self.qin = qin
        self.worker_list = worker_list
        self.cur_w_idx = 0
        self.policy = policy
        

    def run(self):
        timestamp('schedule', 'start')
        
        job_list = []
        while True:
            # Get request           
            i, agent, task_name, data_name, num_layers = self.qin.get()
            job_list.append([i, agent, task_name, data_name, num_layers])
            timestamp('schedule', 'get_request')
            if len(job_list) == len(self.model_list):
                break
        job_list.sort(key=lambda x: x[0])
        job_list = [x[1:] for x in job_list]
        print(job_list)
        
        # run a job on the specified worker
        
        def run_job(job, worker_tuple, memory_threshold):
            if job is None:
                return
            agent, task_name, data_name, num_layers = job[0], job[1], job[2], job[3]
            new_pipe, _, p_mem_parent = worker_tuple

            task_info = TaskInfo(task_name)
            timestamp('[JOB][PEAK_MEMORY]', f'{task_name}: {task_info.get_peak_memory()}')

            model_name = []
            for model in self.model_list:
                if model[0] == task_name and model[1] == data_name and model[2] == num_layers:
                    model_name = model
            p_mem_parent.send(memory_threshold)
            new_pipe.send((agent, model_name))
            timestamp('schedule', 'notify_new_worker')
            
        # run the first job in the queue
        #first_job = job_list.pop(0)
        
        # get the co-locate model
        pkl =  "./model_0806.pkl"
        if self.policy == '0':
            model_schedule = Schedule0(job_list, pkl) 
        elif self.policy == '1':
            model_schedule = Schedule1(job_list, pkl) 
        elif self.policy == '2':
            model_schedule = Schedule2(job_list, pkl)
        elif self.policy == '3':
            model_schedule = Schedule3(job_list, pkl)
        elif self.policy == 'recompute0':
            model_schedule = ScheduleRecompute0(job_list, pkl)
        elif self.policy == 'recompute1':
            model_schedule = ScheduleRecompute1(job_list, pkl)
        elif self.policy == 'recompute2':
            model_schedule = ScheduleRecompute2(job_list, pkl)
        elif self.policy == 'simple':
            model_schedule = ScheduleSimple(job_list, pkl)
        elif self.policy == 'lucid':
            model_schedule = ScheduleLucid(job_list)
        elif self.policy == 'recompute0_static':
            model_schedule = ScheduleRecompute0_static(job_list, pkl)
        elif self.policy == 'recompute0_pair_nocost':
            model_schedule = ScheduleRecompute0_pair_nocost(job_list, pkl)
        elif self.policy == 'recompute0_nocost':
            model_schedule = ScheduleRecompute0_nocost(job_list, pkl)
        elif self.policy == '0_no_inter':
            model_schedule = Schedule0NoInterference(job_list, pkl)
        elif self.policy == '1_no_inter':
            model_schedule = Schedule1NoInterference(job_list, pkl)
        elif self.policy == '2_no_inter':
            model_schedule = Schedule2NoInterference(job_list, pkl)
        elif self.policy == '2_no_recomp':
            model_schedule = Schedule2NoRecomputation(job_list, pkl)    
        else:
            raise NotImplementedError()
        #model_schedule = Schedule(job_list) 

        first_job, job_list = model_schedule.get_first_job()

        timestamp('schedule', 'run first job')
        self.cur_w_idx += 1

        first_job, co_job, job_list, memory_1, memory_2 = model_schedule.get_co_job(first_job)
        timestamp('schedule', 'run first co-job')
        run_job(first_job, self.worker_list[0], memory_1)
        run_job(co_job, self.worker_list[1], memory_2)
        self.cur_w_idx += 1

        #current_jobs = ((first_job, self.worker_list[0]), (co_job, self.worker_list[1]))
        current_jobs = {
            self.worker_list[0][0].fileno(): (first_job, self.worker_list[0]),
            self.worker_list[1][0].fileno(): (co_job, self.worker_list[1])
        }

        epoll = select.epoll()
        epoll.register(self.worker_list[0][0].fileno(), select.EPOLLIN)
        epoll.register(self.worker_list[1][0].fileno(), select.EPOLLIN)

        #w_idx = 0
        #while len(job_list) != 0: 
        #    while True:
        #        if not self.worker_list[w_idx%2][0].poll():
        #            continue
        # while len(job_list) != 0:
        while not model_schedule.finished():
            events = epoll.poll(maxevents=1)
            fd, event = events[0]

            if event & select.EPOLLIN:
        #        fd = self.worker_list[w_idx%2][0].fileno()
                finished_job, finished_worker = current_jobs.pop(fd)
                pipe = finished_worker[0]
                res = pipe.recv()
                timestamp('schedule', 'a job is finished')
                running_job, running_worker = list(current_jobs.values())[0]
                while model_schedule.only_not_ready_job():
                    time.sleep(1)
                    print('[WAITING] waiting for job ready')
                    pass
                if running_job is None:
                    first_job, job_list = model_schedule.get_first_job()
                    first_job, new_co_job, job_list, memory_1, memory_2 = \
                        model_schedule.get_co_job(first_job)
                    print(first_job, new_co_job)
                    
                    current_jobs = {
                        finished_worker[0].fileno(): (first_job, finished_worker),
                        running_worker[0].fileno(): (new_co_job, running_worker)
                    }
                    run_job(first_job, finished_worker, memory_1)
                    run_job(new_co_job, running_worker, memory_2)
                else:
                    first_job, new_co_job, job_list, memory_1, memory_2 = \
                        model_schedule.get_co_job(running_job) 
                    running_worker[2].send(memory_1)
                    current_jobs[fd] = (new_co_job, finished_worker)
                    run_job(new_co_job, finished_worker, memory_2)
                    
        for i in range(len(self.worker_list)):
            list(current_jobs.values())[i][1][0].send([None, None])
            list(current_jobs.values())[i][1][1].join()
        #ppid = os.getppid()
        #subprocess.run([f'kill -9 {ppid}'], shell=True)
        if self.policy == '2':
            inference_overhead = model_schedule.predict_model.time
            recomputation_overhead = model_schedule.recomputation.time
            search_overhead = get_schedule_time() - inference_overhead - recomputation_overhead
            timestamp('[SCHEDULE_OVERHEAD]', f'inference {inference_overhead}')
            timestamp('[SCHEDULE_OVERHEAD]', f'recomputation {recomputation_overhead}')
            timestamp('[SCHEDULE_OVERHEAD]', f'search {search_overhead}')
        exit()
        
        
        


            
        
        #w_idx = 0
        #while len(job_list) != 0: 
        #    while True:
        #        w_idx %= len(self.worker_list)
        #        new_pipe, _, p_mem = self.worker_list[w_idx]
        #        # Recv response
        #        if new_pipe.poll():
        #            if new_pipe == current_jobs[0][1][0]:
        #                free_job = current_jobs[0]
        #                running_job = current_jobs[1]
        #            else:
        #                running_job = current_jobs[0]
        #                free_job = current_jobs[1]
        #            res = new_pipe.recv()
        #            timestamp('schedule', 'a job is finished')
        #            new_co_job, job_list, memory_1, memory_2 = model_schedule.get_co_job(running_job[0])
        #            #new_co_job, job_list = model_schedule.get_co_job()
        #            timestamp('schedule', 'run a new co-job')
        #            run_job(new_co_job, free_job[1], memory_2)
        #            running_job[1][2].send(memory_1)
        #            current_jobs = (running_job, (new_co_job, free_job[1]))
        #            break
        #        w_idx += 1


        # monitor jobs, schedule in time once a job is finished
        #w_idx = 0
        #while(len(job_list)!=0):
        #    while True:
        #        w_idx %= len(self.worker_list)
        #        new_pipe, _ = self.worker_list[w_idx]
        #        # Recv response
        #        if new_pipe.poll():
        #            res = new_pipe.recv()
        #            timestamp('schedule', 'a job is finished')
        #            new_co_job, job_list = model_schedule.get_co_job()
        #            timestamp('schedule', 'run a new co-job')
        #            run_job(new_co_job, w_idx)
        #            break
        #        w_idx += 1
