import os

if not os.path.exists('gc_num.txt'):
    with open('gc_num.txt', 'w') as f:
        gc_nums = (list(range(4, 15)))
        gc_nums = ','.join([str(x) for x in gc_nums])
        f.write(gc_nums)

if not os.path.exists('memory_threshold.txt'):
    with open('memory_threshold.txt', 'w') as f:
        f.write('7,4')

with open('gc_num.txt', 'r') as f:
    gc_nums = f.read().split(',')
    gc_nums = [int(x) for x in gc_nums]
with open('memory_threshold.txt', 'r') as f:
    memory_thresholds = f.read().split(',') 
    max_mem = float(memory_thresholds[0])
    min_mem = float(memory_thresholds[1])
min_fit_mem = 10
import subprocess
for i in range(5):
    new_gc_nums = []
    mem = (max_mem + min_mem)/2
    for gc_num in gc_nums:
        res = subprocess.run([f'python3 vgg_test.py {mem} {gc_num}'], shell=True)
        if res.returncode == 0:
            new_gc_nums.append(gc_num)  
    if len(new_gc_nums) != 0:
        min_fit_mem = mem 
        max_mem = mem
        gc_nums = new_gc_nums
        print(f'fit mem {mem} success with {new_gc_nums}')
    else:
        min_mem = mem 
        print(f'fit mem {mem} fail')



with open('gc_num_new.txt', 'w') as f:
    f.write(','.join([str(x) for x in gc_nums]))

with open('fit_mem.txt', 'a') as f:
    f.write(f'{min_fit_mem}, {max_mem}, {min_mem}\n')