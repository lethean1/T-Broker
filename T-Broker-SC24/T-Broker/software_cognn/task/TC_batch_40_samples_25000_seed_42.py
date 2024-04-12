
from task.TC_base import import_fn
def is_transformer():
    pass
def import_parameters(*args):
    return 0, 20.0 * (1024**3)
def import_task(*args):
    return import_fn( **{
        "BATCH_SIZE": 40,
        "SEED": 42,
        "SAMPLE": 25000,
    })
                