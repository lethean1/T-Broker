from task.transformer_loader import TransformerLoader
from task.question_answering_main_qa_beam_search import main

def import_transformer_task(pipe,name):
    runner = TransformerLoader(main,pipe,name)
    return runner