import json
from task_info_table import task_info_table
### Class


task_data = task_info_table
# todo: load task_data from local file

### Class
class TaskInfo():
    def __init__(self, task_name):
        """ """
        self.task_name = task_name

    def get_peak_memory(self):
        return task_data[self.task_name]['peak_memory']
    
    def get_memory_threshold(self):
        return task_data[self.task_name]['memory_threshold']
        
    def get_metrics(self):
        return task_data[self.task_name]['metrics']
        
    def get_module_info(self):
        return task_data[self.task_name]['module_info']
        
    def get_training_time(self):
        return task_data[self.task_name]['training_time']