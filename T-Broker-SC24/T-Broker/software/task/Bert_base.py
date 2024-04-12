from task.question_answering_main_qa import main
from task.transformer_loader import TransformerLoader

def import_fn(pipe,name, **kwargs):
    BATCH_SIZE=kwargs['BATCH_SIZE']
    SAMPLE=kwargs['SAMPLE']
    WARMUP=kwargs['WARMUP']
    SEED=kwargs['SEED']
    BUFFER=2
    runner = TransformerLoader(main, pipe,name,[
"--model_name_or_path","bert-base-uncased",
"--dataset_name","squad",
"--do_train",
"--per_device_train_batch_size",f"{BATCH_SIZE}",
"--learning_rate","3e-5",
"--num_train_epochs","1",
"--doc_stride","128",
"--dynamic_checkpoint",
f"--max_train_samples",f"{SAMPLE}",
f"--warmup_iters",f"{WARMUP}",
f"--memory_buffer",f"{BUFFER}",
f"--data_seed",f"{SEED}",
"--memory_threshold","31",
"--output_dir","/tmp/test_squad/",
"--overwrite_output_dir"
    ])
    return runner