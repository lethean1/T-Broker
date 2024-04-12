import os
import sys
import time
from queue import Queue

import torch
import torch.multiprocessing as mp

from cognn.frontend_tcp import FrontendTcpThd
from cognn.frontend_schedule import FrontendScheduleThd
from cognn.worker import WorkerProc
from cognn.memory_notifier import MemoryNotifierThd
from util.util import timestamp, TcpAgent, TcpServer
from cognn.server import ServerThd

def main():
    timestamp('frontend', 'start')

    # Load model list (task & data)
    model_list_file_name = sys.argv[1]
    num_workers = int(sys.argv[2])
    model_list = []
    with open(model_list_file_name) as f:
        for line in f.readlines():
            if len(line.split()) != 3:
                continue
            model_list.append([line.split()[0], line.split()[1], line.split()[2]])


    # Warm up CUDA
    timestamp('frontend', 'warmup begin')
    #torch.randn(1024, device='cuda')
    timestamp('frontend', 'warmup end')

    # Create workers
    policy = sys.argv[4]
    is_combomc = (policy != 'lucid')
    
    has_mps = os.system("echo get_server_list | nvidia-cuda-mps-control")
    if has_mps != 0 and is_combomc:
        print("MPS is OFF but we are running combomc!")
        exit(-1)
    elif has_mps == 0 and not is_combomc:
        print("MPS is ON but we are NOT running combomc!")
        exit(-1)
    print("MPS check passed!", flush=True)
    
    worker_list = []
    p_mem_worker_list = []
    for i in range(num_workers):
        p_parent, p_child = mp.Pipe()
        p_mem_parent, p_mem_child = mp.Pipe()
        p_mem_worker_list.append(p_mem_parent)
        worker = WorkerProc(model_list, p_child, p_mem_child, i, is_combomc)
        worker.start()
        worker_list.append((p_parent, worker, p_mem_parent))
        timestamp('frontend', 'create_worker')
    p_mem_notifier_ctr_parent, p_mem_notifier_ctr_child = mp.Pipe() 
    #t_mem = MemoryNotifierThd(p_mem_worker_list, p_mem_notifier_ctr_child)
    #t_mem.start()


    # Create request queue and scheduler thread
    requests_queue = Queue()
    t_sch = FrontendScheduleThd(model_list, requests_queue, worker_list, policy)
    t_sch.start()
    timestamp('frontend', 'start_schedule')

    # Accept connections
    #server = TcpServer('localhost', 12345)
    #timestamp('tcp', 'listen')
    #while True:
    #    conn, _ = server.accept()
    #    agent = TcpAgent(conn)
    #    timestamp('tcp', 'connected')
    #    t_tcp = FrontendTcpThd(requests_queue, agent)
    #    t_tcp.start()
    t_server = ServerThd(requests_queue)
    t_server.start()
    # Wait for end
    t_sch.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
