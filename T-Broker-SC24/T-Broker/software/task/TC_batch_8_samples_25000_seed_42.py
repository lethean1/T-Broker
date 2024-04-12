
from task.TC_base import import_fn
def import_transformer_task(pipe,name,is_combomc=False):
    return import_fn(pipe, name, **{
        "BATCH_SIZE": 8,
        "SEED": 42,
        "SAMPLE": 25000,
        "WARMUP": 20 if is_combomc else 0,
    })
                