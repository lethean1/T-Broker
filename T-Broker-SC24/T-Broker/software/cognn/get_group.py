from cognn.task_info_table import task_info_table 
for key in task_info_table:
    if task_info_table[key]['memory_threshold'] > 20:
        print(key)
