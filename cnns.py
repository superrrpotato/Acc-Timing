import torch
import torch.nn as nn
import layers.linear as linear
import layers.FC as FC
import global_v as glv
import Neuron as f

class Network(nn.Module):
    def __init__(self, network_config, layers_config, input_shape):
        super(Network, self).__init__()
        self.layers = []
        self.network_config = network_config
        self.layers_config = layers_config
        parameters = []
        print("Network Structure:")
        for key in layers_config:
            c = layers_config[key]
            if c['type'] == 'linear':
                self.layers.append(linear.LinearLayer(c, key))
                self.layers[-1].to(glv.device)
                parameters = parameters + self.layers[-1].get_parameters()
            elif c['type'] == 'FC':
                self.layers.append(FC.FCLayer(c, key))
                self.layers[-1].to(glv.device)
                parameters = parameters + self.layers[-1].get_parameters()
            else:
                raise Exception('Undefined layer type. It is:\
                {}'.format(c['type']))
        """
        if network_config["dataset"]=="XOR":
            self.scale = torch.tensor(1.,requires_grad=True, device=glv.device)
            self.bias = torch.tensor(0., requires_grad=True, device=glv.device)
            self.scale = nn.Parameter(self.scale, requires_grad=True)
            self.bias = nn.Parameter(self.bias, requires_grad=True)
            parameters.append(self.scale)
            parameters.append(self.bias)
        """
        self.my_parameters = nn.ParameterList(parameters)
        print("-----------------------------------------")

    def forward(self, spike_input, is_train):
        spikes = f.psp(spike_input, self.network_config)
        skip_spikes = {}
        assert self.network_config['model'] == "LIF"
        for i in range(len(self.layers)):
            if self.layers[i].type == "dropout":
                if is_train:
                    spikes = self.layers[i](spikes)
            elif self.network_config["rule"] == "ATBP":
                spikes = self.layers[i].forward_pass(spikes)
            else:
                raise Exception('Unrecognized rule type. It is:\
                {}'.format(self.network_config['rule']))
        """
        if self.network_config['dataset'] == 'XOR':
            spikes = spikes * self.scale + self.bias
            print("scale: %3.4f, bias: %3.4f"%(self.scale, self.bias))
        """
        return spikes
    def get_parameters(self):
        return self.my_parameters
    def weight_clipper(self):
        for l in self.layers:
            if l.name in ["linear","conv"]:
                l.weight_clipper()
