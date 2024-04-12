
from task.transformer_loader import TransformerLoader
from task.translation_main import main

def import_fn(**kwargs):
    BATCH_SIZE=kwargs['BATCH_SIZE']
    SAMPLE=kwargs['SAMPLE']
    BUFFER=2
    SEED=kwargs['SEED']
    WARMUP=0
    runner = TransformerLoader(main,"",[
"--model_name_or_path","/home/sqx/t5-base",
"--do_train",
"--source_lang","en_XX",
"--target_lang","fr_XX",
"--dataset_name" ,"un_pc",
"--dataset_config_name", "en-fr",
"--output_dir","/tmp/t5-base-un",
"--source_prefix", "'translate English to French: '",
f"--per_device_train_batch_size={BATCH_SIZE}",
f"--max_train_samples",f"{SAMPLE}",
"--max_source_length","512",
"--max_target_length","512",
"--num_train_epochs", "1",
"--overwrite_output_dir",
"--predict_with_generate",
f"--warmup_iters",f"{WARMUP}",
"--memory_threshold", "31",
f"--memory_buffer",f"{BUFFER}",
f"--data_seed",f"{SEED}"
    ])
    return runner.model, runner.do_train, []
    return runner
