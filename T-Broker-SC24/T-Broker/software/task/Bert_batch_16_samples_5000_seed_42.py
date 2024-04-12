
from task.Bert_base import import_fn
from task.transformer_loader import TransformerLoader

def import_transformer_task(pipe,name,is_combomc=False):
    return import_fn(pipe, name, **{
        "BATCH_SIZE": 16,
        "SEED": 42,
        "SAMPLE": 5000,
        "WARMUP": 20 if is_combomc else 0,
    })
                