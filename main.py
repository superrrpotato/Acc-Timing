import torch
import global_v as glv
from network_parser import parse
from datasets import loadMNIST, loadXOR
from utils import learningStats, EarlyStopping
from datetime import datetime
import cnns
import argparse
import loss
import logging

min_loss = 1000

def train(network, trainloader, opti, epoch, states, network_config,\
        layers_config, err):
    global min_loss
    logging.info('\nEpoch: %d', epoch)
    train_loss = correct = total = 0
    n_steps = network_config['n_steps']
#    n_class = network_config['n_class']
    batch_size = network_config['batch_size']
    time = datetime.now()
    des_str = "Training @ epoch " + str(epoch)
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        if network_config["rule"] == "ATBP":
            if len(inputs.shape) < 5:
                inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)
            labels = labels.to(glv.device)
            inputs = inputs.to(glv.device)
            inputs.type(glv.dtype)
            outputs = network.forward(inputs, True)
            if network_config['loss'] == "average":
                target = labels
                loss = err.average(outputs, target)
            opti.zero_grad()
            loss.backward()
            opti.step()
            network.weight_clipper()
            train_loss += torch.sum(loss).item()
            total += len(labels)
        else:
            raise Exception('Unrecognized rule name.')
        states.training.numSamples = total
        states.training.lossSum = train_loss
        states.print(epoch, batch_idx, (datetime.now()-time).total_seconds(),\
                opti.param_groups[0]['lr'])
    total_loss = train_loss/total
    if total_loss < min_loss:
        min_loss = total_loss
    logging.info("Training Loss:  %.3f (%.3f)\n", total_loss, min_loss)

def test(network, testloader, epoch, states, network_config, layers_config,\
        error, early_stopping):
    global best_acc
    global best_epoch
    total = test_loss = 0
    n_steps = network_config['n_steps']
    time = datetime.now()
    des_str = "Testing @ epoch " + str(epoch)
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            if network_config["rule"] == "ATBP":
                if len(inputs.shape) < 5:
                    inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)
                # forward pass
                labels = labels.to(glv.device)
                inputs = inputs.to(glv.device)
                outputs = network.forward(inputs, False)
                if network_config['loss'] == "average":
                    target = labels
                    loss = error.average(outputs,\
                            target).detach().cpu()
                total += len(labels)
                test_loss += torch.sum(loss).item()
            else:
                raise Exception('Unrecognized rule name.')

            states.testing.numSamples = total
            states.testing.lossSum = test_loss
            states.print(epoch, batch_idx, (datetime.now() - time).total_seconds())
    total_loss = test_loss/total
    score = - total_loss
    early_stopping(score, network, epoch)

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
    elif params['Network']['dataset'] == "XOR":
        train_loader, test_loader = loadXOR.get_XOR(params['Network'])
    else:
        raise Exception('Unrecognized dataset name.')
    logging.info("dataset loaded")
    net = cnns.Network(params['Network'], params['Layers'],\
            list(train_loader.dataset[0][0].shape)).to(glv.device)
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
    error = loss.SpikeLoss(params['Network']).to(glv.device)
    optimizer = torch.optim.AdamW(net.get_parameters(),\
            lr=params['Network']['lr'], betas=(0.9, 0.999))
    best_acc = 0
    best_epoch = 0
    l_states = learningStats()
    early_stopping = EarlyStopping()
    decayRate = params['Network']['lr_dacay']
    my_lr_scheduler =\
    torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,\
            gamma=decayRate)
    for e in range(params['Network']['epochs']):
        l_states.training.reset()
        train(net, train_loader, optimizer, e, l_states,\
                params['Network'], params['Layers'], error)
        l_states.training.update()
        my_lr_scheduler.step()
        l_states.testing.reset()
        test(net, test_loader, e, l_states, params['Network'],\
                params['Layers'], error, early_stopping)
        l_states.testing.update()
        if early_stopping.early_stop:
            break
    l_states.plot(True)
