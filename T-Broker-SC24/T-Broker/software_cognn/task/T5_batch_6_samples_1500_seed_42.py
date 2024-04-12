
from task.T5_base import import_fn
def is_transformer():
    pass
def import_parameters(*args):
    return 0, 13.0 * (1024**3)
def import_task(*args):
    return import_fn( **{
        "BATCH_SIZE": 6,
        "SEED": 42,
        "SAMPLE": 1500,
    })
                