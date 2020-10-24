# Acc-Timing
Accurate timing adjustment for Spiking Neural Network
Design the architecture firstly:
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
