
from task.Roberta_base import import_fn
def import_transformer_task(pipe,name,is_combomc=False):
    return import_fn(pipe, name, **{
        "BATCH_SIZE": 16,
        "SEED": 42,
        "SAMPLE": 8000,
        "WARMUP": 20 if is_combomc else 0,
    })
                