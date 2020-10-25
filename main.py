import torch
from network_parser import parse
from datasets import loadMNIST, loadXOR
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
        data_path = os.path.expanduser(params['Network']['data_path'])
        train_loader, test_loader = loadXOR.get_XOR(data_path,\
                params['Network'])
    else:
        raise Exception('Unrecognized dataset name.')
    logging.info("dataset loaded")

