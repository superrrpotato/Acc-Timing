import torch
import Neuron as Neuron
import torch.nn.functional as f
import global_v as glv


def psp(inputs, network_config):
    shape = inputs.shape
    n_steps = network_config['n_steps']
    tau_s = network_config['tau_s']
    syn = torch.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=glv.dtype, device=glv.device)
    syns = torch.zeros((shape[0], shape[1], shape[2], shape[3], n_steps), dtype=glv.dtype, device=glv.device)
    for t in range(n_steps):
        syn = syn - syn / tau_s + inputs[..., t]
        syns[..., t] = syn / tau_s
    return syns


class SpikeLoss(torch.nn.Module):
    """
    This class defines different spike based loss modules that can be used to optimize the SNN.
    """
    def __init__(self, network_config):
        super(SpikeLoss, self).__init__()
        self.network_config = network_config
        self.criterion = torch.nn.CrossEntropyLoss()
    def spike_kernel(self, outputs, target):
        return 1 / 2 * torch.sum((outputs - target) ** 2)
    def average(self, outputs, target):
        return 1 / 2 * torch.sum((torch.sum(outputs,(1, 2, 3, 4))\
                - target) **2)
