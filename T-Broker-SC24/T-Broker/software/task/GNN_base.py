import importlib
def is_gnn():
    pass
def import_task(task_name):
    gnn, _0, dataset, _1, _2, num_layers, epochs = task_name.split('_')
    epochs = int(epochs)
    num_layers = int(num_layers)
    gnn_module = importlib.import_module(f'task.{gnn}_training')
    return gnn_module.import_task(dataset, num_layers, epochs)
def import_model(task_name, data_name, num_layers):
    gnn, _0, dataset, _1, _2, num_layers, epochs = task_name.split('_')
    epochs = int(epochs)
    num_layers = int(num_layers)
    gnn_module = importlib.import_module(f'task.{gnn}_training')
    return gnn_module.import_model(data_name, num_layers)
def import_parameters(task_name, data, num_layers):
    gnn, _0, dataset, _1, _2, num_layers, epochs = task_name.split('_')
    epochs = int(epochs)
    num_layers = int(num_layers)
    gnn_module = importlib.import_module(f'task.{gnn}_training')
    return gnn_module.import_parameters(data, num_layers)
def import_func(task_name):
    gnn, _0, dataset, _1, _2, num_layers, epochs = task_name.split('_')
    epochs = int(epochs)
    num_layers = int(num_layers)
    gnn_module = importlib.import_module(f'task.{gnn}_training')
    model, func, _ = import_task(task_name)
    model, data = import_model(task_name, dataset, num_layers)
    def train():
        return func(model, data)
    return train