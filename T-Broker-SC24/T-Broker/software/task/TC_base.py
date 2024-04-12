
from task.transformer_loader import TransformerLoader
from task.text_classification_main_glue import main

def import_fn(pipe, name, **kwargs):
    BATCH_SIZE=kwargs['BATCH_SIZE']
    SAMPLE=kwargs['SAMPLE']
    BUFFER=2
    SEED=kwargs['SEED']
    WARMUP=kwargs['WARMUP']
    runner = TransformerLoader(main,pipe, name,[
  "--model_name_or_path", "bert-base-cased", 
  "--task_name", f"qqp", 
  "--do_train", 
  "--per_device_train_batch_size", f"{BATCH_SIZE}", 
  "--num_train_epochs", "1", 
  "--learning_rate", "2e-5", 
  "--output_dir", "/tmp/$TASK_NAME/", 
  "--num_train_epochs", "1", 
  "--max_train_samples", f"{SAMPLE}", 
  "--dynamic_checkpoint", 
  "--warmup_iters", f"{WARMUP}", 
  "--memory_buffer", f"{BUFFER}", 
  "--memory_threshold", f"31", 
  "--data_seed", f"{SEED}", 
  "--overwrite_output_dir",
    ])
    return runner
