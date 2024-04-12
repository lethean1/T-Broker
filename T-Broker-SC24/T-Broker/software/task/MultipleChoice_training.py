from task.MultipleChoice import main
from task.transformer_loader import TransformerLoader

def import_transformer_task(pipe,name):
    runner = TransformerLoader(main, pipe,name)
    return runner