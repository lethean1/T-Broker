import threading
from multiprocess.connection import Connection
import time

class MemoryNotifierThd(threading.Thread):
    def __init__(self, p_to_worker_list, p_from_main):
        super(MemoryNotifierThd, self).__init__()
        self.p_to_worker_list = p_to_worker_list
        self.p_from_main = p_from_main
    
    def run(self):
        mem = [4,10]
        i = 0
        while True:
            #worker_id, memory_threshold = self.p_from_main.recv()
            #self.p_to_worker_list[worker_id].send(memory_threshold)
            self.p_to_worker_list[0].send(mem[i%2])
            print(f'change to {mem[i%2]}')
            time.sleep(30)
            i += 1