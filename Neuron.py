import torch
import global_v as glv

class Neuron(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, network_config):
        shape = inputs.shape
        n_steps = glv.n_steps #network_config['n_steps']
        theta_m = glv.theta_m #1/network_config['tau_m']
        theta_s = glv.theta_s #1/network_config['tau_s']
        threshold = glv.threshold #network_config['threshold']
        mem = torch.zeros((shape[0], shape[1], shape[2], shape[3]),\
                dtype=glv.dtype, device=glv.device)
        syn = torch.zeros((shape[0], shape[1], shape[2], shape[3]),\
                dtype=glv.dtype, device=glv.device)
        mems = []
        mem_updates = []
        outputs = []
        syns = []
        for t in range(n_steps):
            mem_update = (-theta_m) * mem + inputs[..., t]
            mem += mem_update
            out = (mem > threshold).type(glv.dtype)
            mems.append(mem)
            mem = mem * (1-out)
            outputs.append(out)
            mem_updates.append(mem_update)
            syn = syn + (out - syn) * theta_s
            syns.append(syn)
        mems = torch.stack(mems, dim = 4)
        mem_updates = torch.stack(mem_updates, dim = 4)
        syns = torch.stack(syns, dim = 4)
        outputs = torch.stack(outputs, dim = 4)
        ctx.save_for_backward(mem_updates, outputs, mems)
        return syns
    @staticmethod
    def backward(ctx, grad_delta):
        (delta_u, outputs, u) = ctx.saved_tensors
        shape = outputs.shape
        n_steps = glv.n_steps

