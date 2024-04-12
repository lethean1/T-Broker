import torch
from torchvision.models import vgg16
from transformers.manager import cast_forward, Manager
import torch.utils.checkpoint as checkpoint
from timeit import default_timer as timer
from torchsummary import summary
import torchsummary
from torchvision import datasets, transforms

import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
root="/home/sqx/combomc/software/task/fashion_mnist/"

batch_size = 128
learning_rate = 0.0002

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

train_dir = './data/train'
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

from torch import nn


class GC_Layer(nn.Module):
    def __init__(self, layers, name_prefix=''):
        super().__init__()
        self.name_prefix=name_prefix
        self.layers = layers
        for i, layer in enumerate(layers):
            self.add_module(f"{name_prefix}layer{i}", layer)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def append(self, layers):
        for i, layer in enumerate(layers):
            self.add_module(f"{self.name_prefix}layer{i}", layer)
        self.layers += layers
        
class VGG_Test(nn.Module):
    def __init__(self, gc_num):
        super().__init__()

        self.iters = 0
        self.vgg = vgg16()
        modules_list = []
        for m in self.vgg.get_submodule('features'):
            modules_list.append(m) 
        modules_list.append(self.vgg.get_submodule('avgpool')) 
        modules_list.append(torch.nn.Flatten())
        for m in self.vgg.get_submodule('classifier'):
            modules_list.append(m) 
        def cast_checkpoint_default(func, *args, **kwargs):
            # casted_args = []
            # for arg in args:
            #     if isinstance(arg, int):
            #         casted_args.append(Variable(torch.Tensor([arg])))
            #     else:
            #         casted_args.append(arg)
            # casted_kwargs = {}
            # for k, v in kwargs.items():
            #     if isinstance(arg, int):
            #         casted_kwargs[k] = Variable(torch.Tensor([v]))
            #     else:
            #         casted_kwargs[k] = v
            # return checkpoint.checkpoint(func, *casted_args, **casted_kwargs)

            def create_custom_forward(module, **kwargs):
                def custom_forward(*inputs):
                    return module(*inputs, **kwargs)
                return custom_forward

            return checkpoint.checkpoint(create_custom_forward(func, **kwargs), *args, preserve_rng_state=False)
        #gc_layers = []
        #for i in range(gc_num):
        #    modules = modules_list[len(modules_list)//gc_num*i:len(modules_list)//gc_num*(i+1)]
        #    gc_layers.append(GC_Layer(modules, f'gc{i}_'))
        #gc_layers[-1].append(modules_list[len(modules_list)//gc_num*gc_num:])

        
        gc_layers = modules_list
        self.layers = gc_layers 
        for i, layer in enumerate(self.layers):
            #if i == 9:
            if False:
                l = i
                old_forward = layer.forward 
                def old_forward_with_timer(*args, **kwargs):
                    torch.cuda.nvtx.range_push(f'iter{self.iters}layer{l}') 
                    ret = old_forward(*args, **kwargs)
                    torch.cuda.nvtx.range_pop()
                    return ret
                    
                def forward(*args, **kwargs):
                    ret = cast_checkpoint_default(old_forward_with_timer, *args, **kwargs) 
                    return ret
                layer.forward = forward
                pass
            self.add_module(f"layer{i}", layer)

    def forward(self, x):
        self.iters += 1
        for layer in self.layers:
            x = layer(x)
        return x
    #@property
    #def gc_layers(self):
    #    modules_list = []
    #    for i, layer in enumerate(self.layers):
    #        modules_list.append(layer)

    #    return modules_list[1:-1]
            
#names = []
#l1 = []
#for i in range(len(modules_list)//2*2):
#    if i % 2 == 0:
#        model.add_module("module{}".format(i), nn.Sequential(
#            modules_list[i],
#            modules_list[i+1]
#        ))    
#        names.append("module{}".format(i))
#for i in range(len(modules_list)//2*2, len(modules_list)):
#    model.add_module("module{}".format(i), modules_list[i])
#    names.append("module{}".format(i))


def test(memory_threshold, gc_num):

    #try:
        torch.cuda.memory.reset_peak_memory_stats()
        torch.cuda.memory.empty_cache()
        model = VGG_Test(gc_num)        
        model = model.cuda()
        manager = Manager()
        manager.set_max_memory_GB(memory_threshold)
        manager.set_input_size((3,224,224))
        #cast_forward(model, "0", manager)
        torch.cuda.set_per_process_memory_fraction(memory_threshold * (1024 ** 3) / torch.cuda.get_device_properties(0).total_memory)

        optimizer = torch.optim.Adam(model.parameters())
        loss_func = torch.nn.CrossEntropyLoss()

        for epoch in range(1):
            #print('epoch {}'.format(epoch + 1))
            # training-----------------------------
            train_loss = 0.
            train_acc = 0.
            for i, (batch_x, batch_y) in enumerate(train_dataloader):
                #torch.cuda.empty_cache()
                st = timer() 
                #torch.cuda.memory.empty_cache()
                batch_x = Variable(batch_x).cuda()
                batch_y = Variable(batch_y).cuda()
                out = model(batch_x)
                loss = loss_func(out, batch_y)
                train_loss += loss.item()
                pred = torch.max(out, 1)[1]
                train_correct = (pred == batch_y).sum()
                train_acc += train_correct.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ed = timer()
                print(f"time: {ed-st}")
                if i >= 20:
                    return True

            print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
                train_datasets)), train_acc / (len(train_datasets))))
    #except:
    #    return False
    #return True

    # evaluation--------------------------------
    #model.eval()
    #eval_loss = 0.
    #eval_acc = 0.
    #for batch_x, batch_y in test_loader:
    #    batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
    #    batch_x = batch_x.cuda()
    #    batch_y = batch_y.cuda()
    #    out = model(batch_x)
    #    loss = loss_func(out, batch_y)
    #    eval_loss += loss.item()
    #    pred = torch.max(out, 1)[1]
    #    num_correct = (pred == batch_y).sum()
    #    eval_acc += num_correct.item()
    #print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    #    test_data)), eval_acc / (len(test_data))))


import sys
if test(float(sys.argv[1]), int(sys.argv[2])):
    print(0)
    exit(0)
else:
    print(-1)
    exit(-1)
#import os
#
#if not os.path.exists('gc_num.txt'):
#    with open('gc_num.txt', 'w') as f:
#        gc_nums = (list(range(4, 30)))
#        gc_nums = ','.join([str(x) for x in gc_nums])
#        f.write(gc_nums)
#
#if not os.path.exists('memory_threshold.txt'):
#    with open('memory_threshold.txt', 'w') as f:
#        f.write('7,4')
#
#with open('gc_num.txt', 'r') as f:
#    gc_nums = f.read().split(',')
#    gc_nums = [int(x) for x in gc_nums]
#with open('memory_threshold.txt', 'r') as f:
#    memory_thresholds = f.read().split(',') 
#    max_mem = float(memory_thresholds[0])
#    min_mem = float(memory_thresholds[1])
#min_fit_mem = 10
#
#for i in range(10):
#    new_gc_nums = []
#    mem = (max_mem + min_mem)/2
#    for gc_num in gc_nums:
#        if test(mem, gc_num):
#            new_gc_nums.append(gc_num)  
#    if len(new_gc_nums) != 0:
#        min_fit_mem = mem 
#        max_mem = mem
#        gc_nums = new_gc_nums
#        print(f'fit mem {mem} success with {new_gc_nums}')
#    else:
#        min_mem = mem 
#        print(f'fit mem {mem} fail')
#
#
#
#with open('gc_num.txt', 'w') as f:
#    f.write(','.join([str(x) for x in gc_nums]))
#
#with open('fit_mem.txt', 'a') as f:
#    f.write(f'{min_fit_mem}, {max_mem}, {min_mem}')