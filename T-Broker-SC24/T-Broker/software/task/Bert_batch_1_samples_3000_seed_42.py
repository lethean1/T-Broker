
from task.Bert_base import import_fn
def import_transformer_task(pipe,name,is_combomc=False):
    return import_fn(pipe, name, **{
        "BATCH_SIZE": 1,
        "SEED": 42,
        "SAMPLE": 3000,
    })
                