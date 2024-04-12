from task.transformer_loader import TransformerLoader
from task.text_classification_main_glue import main

def import_transformer_task(pipe, name):
    runner = TransformerLoader(main,pipe, name)
    return runner
