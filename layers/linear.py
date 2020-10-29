import torch
import torch.nn as nn
import torch.nn.functional as f
import Neuron as Neuron
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import global_v as glv

class LinearLayer(nn.Linear):
    def __init__(self, config, name):
        n_inputs = config['n_inputs']
        n_outputs = config['n_outputs']
        assert(type(n_inputs) == int)
        assert(type(n_outputs) == int)
        super(LinearLayer, self).__init__(n_inputs, n_outputs, bias=False)
        nn.init.normal_(self.weight)
#        nn.init.zeros_(self.bias)
        self.weight = torch.nn.Parameter(self.weight, requires_grad=True)
#        self.bias = torch.nn.Parameter(self.bias, requires_grad=True)
        print("linear")
        print(name)
        print("input shape:", [glv.batch_size, n_inputs, 1, 1, glv.n_steps])
        print("weight shape: ", list(self.weight.shape)[::-1])
        print("output shape:", [glv.batch_size, n_outputs, 1, 1, glv.n_steps])
        print("-----------------------------------------")

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3],\
                x.shape[4])
        x = x.transpose(1, 2)
        y = f.linear(x, self.weight, self.bias)
        y = y.transpose(1, 2)
        y = y.view(y.shape[0], y.shape[1], 1, 1, y.shape[2])
        return y
    def forward_pass(self, x):
        y = self.forward(x)
        y = Neuron.Neuron.apply(y)
        return y
    def get_parameters(self):
        return self.weight
    def weight_clipper(self):
        w = self.weight.data
        w = w.clamp(-4, 4)
        self.weight.data = w
