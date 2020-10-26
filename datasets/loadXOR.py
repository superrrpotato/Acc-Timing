import torch

class XORDataset(data.Dataset):
    def __init__(self, train=True):
        self.train = train
    def __getitem__(self, index):
        data = torch.rand(2)
        (x, y) = data
        def sigmoid(x,t=0.15):return 1/(1+np.exp(-x/t))
        def XOR(x,y):return -4*(sigmoid(x)-0.5)*(sigmoid(y)-0.5)
        label = XOR(x,y)
        return data, label
    def __len__(self):
        return 5000 if self.train else 1000

def get_XOR(network_config):
    print("loading XOR")
    batch_size = network_config['batch_size']
    trainset = XORDataset(train=True)
    testset = XORDataset(train=False)
    trainloader = torch.utils.data.DataLoader(trainset,\
            batch_size=batch_size, shuule=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset,\
            batch_size=batch_size, shuule=False, num_workers=4)
    return trainloader, testloader
