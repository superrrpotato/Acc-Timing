import torch
from network_parser import parse
from utils import aboutCudaDevices

def init(params):
    global device, dtype, n_steps, theta_m, theta_s, threshold, partial_a
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda.init()
        c_device = aboutCudaDevices()
        print(c_device.info())
        print("selected device: ", device)
    dtype = torch.float32
    network_config = params['Network']
    n_steps = network_config['n_steps']
    theta_m = 1/network_config['tau_m']
    theta_s = 1/network_config['tau_s']
    threshold = network_config['threshold']
    partial_a = torch.zeros((1, 1, 1, 1, n_steps, n_steps),\
            dtype=dtype, device = device)
    tau_s = network_config['tau_s']
    for t in range(n_steps):
        if t > 0:
            partial_a[..., t] = partial_a[..., t - 1] - partial_a[..., t - 1]\
            /tau_s
        partial_a[..., t, t] = 1/tau_s


