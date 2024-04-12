
from task.T5_base import import_fn
def import_transformer_task(pipe,name,is_combomc=False):
    return import_fn(pipe, name, **{
        "BATCH_SIZE": 7,
        "SEED": 42,
        "SAMPLE": 1500,
        "WARMUP": 20 if is_combomc else 0,
    })
