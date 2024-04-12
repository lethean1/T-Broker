
from task.Roberta_base import import_fn
def is_transformer():
    pass
def import_parameters(*args):
    return 0, 12.5 * (1024**3)
def import_task(*args):
    return import_fn( **{
        "BATCH_SIZE": 16,
        "SEED": 42,
        "SAMPLE": 8000,
    })
                