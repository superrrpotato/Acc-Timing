import torch
import torch.nn as nn
import torch.nn.functional as f
import Neuron as Neuron
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import global_v as glv

class FCLayer(nn.Linear):
    def __init__(self, Network_cofig, config, name):
        n_inputs = config['n_inputs'] * glv.n_steps
        n_outputs = config['n_outputs']
        self.config = config
        self.name = 'FC'
        assert(type(n_inputs) == int)
        assert(type(n_outputs) == int)
        super(LinearLayer, self).__init__(n_inputs, n_outputs, bias=True)
        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)+0.1
        self.weight = torch.nn.Parameter(self.weight, requires_grad=True)
        self.bias = torch.nn.Parameter(self.bias, requires_grad=True)
        print(self.bias.shape)

        print("FC")
        print(name)
        print("input shape:", [glv.batch_size, n_inputs])
        print("weight shape: ", list(self.weight.shape)[::-1])
        print("output shape:", [glv.batch_size, n_outputs])
        print("-----------------------------------------")

    def forward_pass(self, x):
        x = x.view(-1)
        y = f.linear(x, self.weight, self.bias)
        return y
    def get_parameters(self):
        return [self.weight, self.bias]
    def weight_clipper(self):
        w = self.weight.data
        w = w.clamp(-4, 4)
        self.weight.data = w

