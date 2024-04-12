from task.transformer_loader import TransformerLoader
from task.translation_main import main

def import_transformer_task(pipe, name):
    runner = TransformerLoader(main, pipe, name)
    return runner
