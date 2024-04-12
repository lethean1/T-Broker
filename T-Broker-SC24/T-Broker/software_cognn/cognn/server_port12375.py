from threading import Thread
from util.util import timestamp, TcpAgent, TcpServer
from frontend_tcp import FrontendTcpThd
class ServerThd(Thread):
    def __init__(self, requests_queue):
        super(ServerThd, self).__init__(daemon=True)
        self.requests_queue = requests_queue
    def run(self):
        
        # Accept connections
        server = TcpServer('localhost', 12375)
        timestamp('tcp', 'listen')
        while True:
            conn, _ = server.accept()
            agent = TcpAgent(conn)
            timestamp('tcp', 'connected')
            t_tcp = FrontendTcpThd(self.requests_queue, agent)
            t_tcp.start()