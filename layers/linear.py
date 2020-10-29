import torch
import torch.nn as nn
import torch.nn.functional as f
import Neuron as Neuron

class LinearLayer(nn.Linear):
    def __init__(self, config, name, in_shape):
        n_inputs = config['n_inputs']
        n_outputs = config['n_outputs']
        self.in_shape = in_shape
        self.out_shape = [n_outputs, 1, 1]
        self.in_spikes = None
        self.out_spikes = None
        assert(type(n_inputs) == int)
        assert(type(n_outputs) == int)
        super(LinearLayer, self).__init__(n_inputs, n_outputs, bias=False)
        nn.init.normal_(self.weight)
#        nn.init.zeros_(self.bias)
        self.weight = torch.nn.Parameter(self.weight, requires_grad=True)
#        self.bias = torch.nn.Parameter(self.bias, requires_grad=True)
        print("linear")
        print(name)
        print(self.in_shape)
        print(self.out_shape)
        print(list(self.weight.shape))
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
