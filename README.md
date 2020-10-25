# Acc-Timing
Accurate timing adjustment for Spiking Neural Network
1.Design the architecture:
Network
    Layers
        Conv
            Forward
                Pytorch conv3d
                +
                Neuron (autograd)
                    Forward
        Linear
            Forward
                Pytorch linear
                +
                Neuron (autograd)
                    Forward
        Pooling
            Forward
        Dropout
            Forward
        BN
            Forward
Neuron (autograd)
    Forward
    Backward

2.Configurations
Network config
Layers config

