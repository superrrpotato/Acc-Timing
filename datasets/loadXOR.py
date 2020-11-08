import torch
from torch.utils.data import Dataset, DataLoader
class XORDataset(Dataset):
    def __init__(self, train=True):
        self.train = train
    def __getitem__(self, index):
        data = 2*(torch.rand(2)-0.5)
        (x, y) = data
        def sigmoid(x,t=0.15):return 1/(1+torch.exp(-x/t))
        def XOR(x,y):return -4*(sigmoid(x)-0.5)*(sigmoid(y)-0.5)
        label = XOR(x,y)
        return data.view(2,1,1), label
    def __len__(self):
        return 500000 if self.train else 100000

def get_XOR(network_config):
    print("loading XOR")
    batch_size = network_config['batch_size']
    trainset = XORDataset(train=True)
    testset = XORDataset(train=False)
    trainloader = DataLoader(trainset, batch_size=batch_size,\
            shuffle=False, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size,\
            shuffle=False, num_workers=4)
    return trainloader, testloader
