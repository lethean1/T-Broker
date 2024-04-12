from task.MultipleChoice import main
from task.transformer_loader import TransformerLoader

def import_fn( **kwargs):
    BATCH_SIZE=kwargs['BATCH_SIZE']
    SAMPLE=kwargs['SAMPLE']
    WARMUP=0
    SEED=kwargs['SEED']
    BUFFER=2
    runner = TransformerLoader(main, "",[
"--model_name_or_path","roberta-base",
"--output_dir","/tmp/test-swag-no-trainer",
"--num_train_epochs","1",
"--do_train",
f"--per_device_train_batch_size", f"{BATCH_SIZE}",
f"--warmup_iters", f"{WARMUP}",
f"--memory_buffer", f"{BUFFER}",
"--memory_threshold","31",
f"--data_seed" ,f"{SEED}",
f"--max_train_samples",f"{SAMPLE}",
"--overwrite_output_dir"
    ])
    return runner.model, runner.do_train, []
    return runner