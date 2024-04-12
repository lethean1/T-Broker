from task.mmdet_base import get_cfg, get_model, mmdet_train

def import_func(cfg, model, datasets, others):
    def train():
        print("mmdet training >>>>>>>>>>>>>>>>>>>>>>")
        mmdet_train(cfg, model, datasets, others)
        print("finish mmdet train")
    return train
def import_model(data, cfg):
    model, datasets = get_model(cfg)
    return model, datasets
def import_mmdet_task(task_name, cfg_path, pipe):
    cfg, others = get_cfg(cfg_path)
    others = tuple(list(others) + [pipe])
    model, datasets = get_model(cfg)
    return import_func(cfg, model, datasets, others)