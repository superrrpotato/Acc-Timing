import torch
from network_parser import parse
from datasets import loadMNIST, loadXOR
from utils import learningStats
import cnns
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', action='store', dest='config',\
            help='The path of config file')
    parser.add_argument('-checkpoint', action='store', dest='checkpoint',\
            help='The path of checkpoint, if use checkpoint')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)
    if args.config is None:
        raise Exception('Unrecognized config file.')
    else:
        config_path = args.config
    logging.basicConfig(filename='result.log', level=logging.INFO)
    logging.info("start parsing settings")
    params = parse(config_path)
    logging.info("finish parsing settings")
    glv.init(params)
    logging.info("start loading dataset")
    if params['Network']['dataset'] == "MNIST":
        data_path = os.path.expanduser(params['Network']['data_path'])
        train_loader, test_loader = loadMNIST.get_mnist(data_path,\
                params['Network'])
    elif params['Network']['dataset'] == "XOR"
        train_loader, test_loader = loadXOR.get_XOR(params['Network'])
    else:
        raise Exception('Unrecognized dataset name.')
    logging.info("dataset loaded")
    net = cnns.Network(params['Network'], params['Layers'],\
            list(train_loader.dataset[0][0].shape)).to(device)
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
    error = loss_f.SpikeLoss(params['Network']).to(device)
    optimizer = torch.optim.AdamW(net.get_parameters(),\
            lr=params['Network']['lr'], betas=(0.9, 0.999))
    best_acc = 0
    best_epoch = 0
    l_states = learningStats()
    for e in range(params['Network']['epochs']):
        l_states.training.reset()
        train(net, train_loader, optimizer, e, l_states,\
                params['Network'], params['Layers'], error)
        l_states.training.update()
        l_states.testing.reset()
        test(net, test_loader, e, l_states, params['Network'],\
                params['Layers'])
        l_states.testing.update()
    logging.info("Best Accuracy: %.3f, at epoch: %d \n", best_acc, best_epoch)
