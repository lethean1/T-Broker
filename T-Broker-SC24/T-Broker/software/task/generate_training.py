import os
with open('name.txt') as f:
    names = f.readlines()
names = [name.strip() for name in names]
for name in names:
    #if not os.path.exists(f'{name}.py'):
    if True:
        with open(f'{name}.py', 'w') as f:
            model_name, _0, batch_size, _1, sample, _2, seed = name.split('_')
            f.write(f'''
from task.{model_name}_base import import_fn
def import_transformer_task(pipe,name):
    return import_fn(pipe, name, **{{
        "BATCH_SIZE": {batch_size},
        "SEED": {seed},
        "SAMPLE": {sample},
    }})
                ''')