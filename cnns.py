import torch
import torch.nn as nn
import layers.linear as linear
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
                self.layers.append(linear.LinearLayer(c, key, input_shape))
                self.layers[-1].to(glv.device)
                input_shape = self.layers[-1].out_shape
                parameters.append(self.layers[-1].get_parameters())
            else:
                raise Exception('Undefined layer type. It is:\
                {}'.format(c['type']))
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
            return spikes
    def get_parameters(self):
        return self.my_parameters
    def weight_clipper(self):
        for l in self.layers:
            if l.name in ["linear","conv"]:
                l.weight_clipper()
