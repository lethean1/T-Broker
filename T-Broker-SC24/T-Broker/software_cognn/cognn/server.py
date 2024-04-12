from threading import Thread
from util.util import timestamp, TcpAgent, TcpServer
from frontend_tcp import FrontendTcpThd
import sys
import subprocess

class ServerThd(Thread):
    def __init__(self, requests_queue):
        super(ServerThd, self).__init__(daemon=True)
        self.requests_queue = requests_queue
    def run(self):
        
        # Accept connections
        server = TcpServer('localhost', int(sys.argv[3]))
        subprocess.run([f'python client.py {sys.argv[1]} {sys.argv[3]} &'], shell=True, cwd='../client')
        timestamp('tcp', 'listen')
        while True:
            conn, _ = server.accept()
            agent = TcpAgent(conn)
            timestamp('tcp', 'connected')
            t_tcp = FrontendTcpThd(self.requests_queue, agent)
            t_tcp.start()