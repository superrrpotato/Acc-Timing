import torch
import global_v as glv


def psp(inputs, network_config):
    shape = inputs.shape
    n_steps = network_config['n_steps']
    tau_s = network_config['tau_s']
    syn = torch.zeros((shape[0], shape[1], shape[2], shape[3]),\
            dtype=glv.dtype, device=glv.device)
    syns = torch.zeros((shape[0], shape[1], shape[2], shape[3], n_steps),\
            dtype=glv.dtype, device=glv.device)
    for t in range(n_steps):
        syn = syn - syn / tau_s + inputs[..., t]
        syns[..., t] = syn / tau_s
    return syns


class Neuron(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        shape = inputs.shape
        n_steps = glv.n_steps
        theta_m = glv.theta_m
        theta_s = glv.theta_s
        threshold = glv.threshold
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
        threshold = glv.threshold
        grad_a = torch.empty_like(delta_u)
        if shape[4] > shape[0] * 10:
            for t in range(n_steps):
                partial_a_inter = glv.partial_a[..., t, :].repeat(shape[0],\
                        shape[1], shape[2], shape[3], 1)
                grad_a[..., t] = torch.sum(partial_a_inter[...,\
                    t:n_steps]*grad_delta[..., t:n_steps], dim=4)
        else:
            mini_batch = 5
            partial_a_inter = glv.partial_a.repeat(mini_batch, shape[1],\
                    shape[2], shape[3], 1, 1)
            for i in range(int(shape[0]/mini_batch)):
                grad_a[i*mini_batch:(i+1)*mini_batch, ...] =\
                    torch.einsum('...ij, ...j -> ...i',partial_a_inter,\
                        grad_delta[i*mini_batch:(i+1)*mini_batch, ...])
        if torch.sum(outputs)/(shape[0] * shape[1] * shape[2] * shape[3] *\
                shape[4]) > 1:
            partial_u = torch.clamp(1 / delta_u, -10, 10) * outputs
            grad = grad_a * partial_u
        else:
            a = 0.2
            f = torch.clamp((-1 * u + threshold) / a, -8, 8)
            f = torch.exp(f)
            f = f / ((1 + f) * (1 + f) * a)
            grad = grad_a * f
        return grad



