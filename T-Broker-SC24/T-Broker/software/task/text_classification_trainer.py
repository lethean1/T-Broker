from typing import DefaultDict
import torch, json
from torch import nn
from tqdm import tqdm
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import Trainer
from transformers.trainer import logger
from transformers.manager import cast_forward, recover_forward, Manager
#from register_hook import register_hook, memory_iter, mark_forward_end

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

def print_shape(inputs):
    for k, v in inputs.items():
        print(f"{k}: {v.shape}")


class CountShape(Trainer):
    def __init__(self, *args, **kwargs):
        training_args = kwargs["args"]
        memory_threshold = training_args.memory_threshold
        self.memory_threshold_pipe = kwargs['pipe']
        kwargs.pop('pipe')
        self.memory_threshold = memory_threshold
        self.memory_buffer = training_args.memory_buffer
        if memory_threshold > 3:
                torch.cuda.set_per_process_memory_fraction(memory_threshold * (1024 ** 3) / torch.cuda.get_device_properties(0).total_memory)
        if kwargs["args"].dynamic_checkpoint:
            warmup_iters = training_args.warmup_iters
            self.dc_manager = Manager(warmup_iters=warmup_iters)
            self.dc_manager.set_max_memory_GB(memory_threshold=memory_threshold-training_args.memory_buffer)
            self.dc_manager.static_strategy = training_args.static_checkpoint
            self.dc_manager.max_input = training_args.max_input_size
            self.dc_manager.min_input = training_args.min_input_size
            cast_forward(kwargs["model"].bert.encoder, "0", self.dc_manager)
        # if training_args.profiling_memory:
        #     model = kwargs["model"].bert
        #     register_hook(model.embeddings)
        #     for layer in model.encoder.layer:
        #         register_hook(layer)
        #     register_hook(model.pooler)
        #     model.pooler.register_forward_hook(mark_forward_end)
        super().__init__(*args, **kwargs)
        self.input_shape = DefaultDict(int)
        self.memory_collect = {}
        self.shape_order = []
        self.profile_memory = kwargs.get("profile_memory", False)


    def training_step(self, model: nn.Module, inputs) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        seq_length = inputs['input_ids'].shape[-1]
        self.input_shape[seq_length] += 1
        self.shape_order.append(seq_length)

        if self.profile_memory:
            torch.cuda.empty_cache()
        torch.cuda.memory.reset_peak_memory_stats()
        new_mem = False
        while self.memory_threshold_pipe.poll():
            self.memory_threshold = self.memory_threshold_pipe.recv()
            new_mem = True
        if new_mem:
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(
                self.memory_threshold * (1024 ** 3)  \
                    / torch.cuda.get_device_properties(0).total_memory)
        self.dc_manager.set_max_memory_GB(
            memory_threshold=
                self.memory_threshold-self.memory_buffer)
        #self.dc_manager.before_model_forward()
        if self.args.dynamic_checkpoint:
            self.dc_manager.set_input_size(seq_length)
        ret = super().training_step(model, inputs)

        # if self.profile_memory:
        if seq_length not in self.memory_collect:
            self.memory_collect[seq_length] = []
        self.memory_collect[seq_length].append(torch.cuda.max_memory_allocated())

        if self.args.dynamic_checkpoint:
            self.dc_manager.after_update()
        # torch.cuda.synchronize()
        # if self.args.profiling_memory and self.state.global_step >= 2:
        #     forward_end = memory_iter.pop("forward_end")
        #     begin_time = min(memory_iter.keys())
        #     tmp = {key - begin_time: value for key, value in memory_iter.items()}
        #     tmp["forward_end"] = forward_end - begin_time
        #     logger.info("memory_iter=" + json.dumps(tmp))
        #     exit(0)
        # if self.args.profiling_memory:
        #     memory_iter.clear()
        return ret
    
    def count_input_size(self):
        train_dataloader = self.get_train_dataloader()
        for inputs in tqdm(train_dataloader):
            seq_length = inputs['input_ids'].shape[-1]
            self.input_shape[seq_length] += 1
            self.shape_order.append(seq_length)
        logger.info("shape_count=" + json.dumps(self.input_shape))
        logger.info("memory_count=" + json.dumps(self.memory_collect))
        logger.info("shape_order=" + json.dumps(self.shape_order))
        exit(0)


    def train(self, *args, **kwargs):
        if hasattr(self.args, "only_input_size") and self.args.only_input_size:
            self.count_input_size()

        ret = super().train(*args, **kwargs)
        if len(self.memory_collect) == 0:
            self.memory_collect[-1] = [torch.cuda.max_memory_allcated()]
        logger.info("shape_count=" + json.dumps(self.input_shape))
        logger.info("memory_count=" + json.dumps(self.memory_collect))
        logger.info("shape_order=" + json.dumps(self.shape_order))
        if self.args.dynamic_checkpoint:
            strategy = {}
            for k in self.dc_manager.cached_strategy:
                strategy[str(k)] = self.dc_manager.cached_strategy[k]
            logger.info("strategy: " + json.dumps(strategy, cls=SetEncoder))
            #logger.info("strategy: " + json.dumps(self.dc_manager.cached_strategy, cls=SetEncoder))
            recover_forward(self.model.bert.encoder)
        return ret
    
    
    def _save_checkpoint(self, model, trial, metrics=None):
        pass
