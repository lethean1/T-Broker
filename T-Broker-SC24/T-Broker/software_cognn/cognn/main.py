import sys
import time
from queue import Queue

import torch
import torch.multiprocessing as mp

from cognn.frontend_tcp import FrontendTcpThd
from cognn.frontend_schedule import FrontendScheduleThd
from cognn.worker import WorkerProc
from util.util import timestamp, TcpAgent, TcpServer
from cognn.server import ServerThd

import os

def main():
    timestamp('frontend', 'start')
    
    has_mps = os.system("echo get_server_list | nvidia-cuda-mps-control")
    if has_mps == 0:
        pass
        print("MPS is ON but we are running cognn!")
        #exit(-1)
    timestamp('frontend', 'mps check')

    # Load model list (task & data)
    model_list_file_name = sys.argv[1]
    num_workers = int(sys.argv[2])
    model_list = []
    with open(model_list_file_name) as f:
        for line in f.readlines():
            if len(line.split()) != 3:
                continue
            model_list.append([line.split()[0], line.split()[1], line.split()[2]])

    # Warm up CUDA and allocate shared cache
    #torch.randn(1024, device='cuda')
    #torch.cuda.allocate_shared_cache()

    # Create workers
    worker_list = []
    for i in range(num_workers):
        p_parent, p_child = mp.Pipe()
        param_trans_parent, param_trans_child = mp.Pipe()
        worker = WorkerProc(model_list, p_child, param_trans_child, i)
        worker.start()
        #torch.cuda.send_shared_cache()
        worker_list.append((p_parent, worker, param_trans_parent))
        timestamp('frontend', 'create_worker')


    # Create request queue and scheduler thread
    requests_queue = Queue()
    t_sch = FrontendScheduleThd(model_list, requests_queue, worker_list)
    t_sch.start()
    timestamp('frontend', 'start_schedule')

    # Accept connections
    #server = TcpServer('localhost', 12355)
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
