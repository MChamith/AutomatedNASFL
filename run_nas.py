import pickle
import random

from gpt_util import initialize_configs, intermediate_configs
from client_process import get_best_config
from torchvision import datasets, transforms
import numpy as np
from non_iid_partition import partition_noniid
# from datasets import load_dataset

def load_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    data_train = datasets.CIFAR10(root='./data/cfar', train=True,
                                  download=True, transform=transform)
    data_test = datasets.CIFAR10(root='./data/cfar', train=False,
                                 download=True, transform=transform)

    return data_train, data_test

# def load_dataset_imgnet():
#     print('loading imagenet')
#     data_train = load_dataset('Maysee/tiny-imagenet', split='train')
#     data_test = load_dataset('Maysee/tiny-imagenet', split='valid')
#     print('data train ' + str(data_train[0]))
#     # data_train = convert_data(data_train)
#     # data_test = convert_data(data_test)
#
#     return data_train, data_test

def iid_partition(dataset, K):
    num_items_per_client = int(len(dataset) / K)
    client_dict = {}
    image_idxs = [i for i in range(len(dataset))]

    for i in range(K):
        client_dict[i] = set(np.random.choice(image_idxs, num_items_per_client, replace=False))
        image_idxs = list(set(image_idxs) - client_dict[i])

    return client_dict


if __name__ == '__main__':
    from multiprocessing import freeze_support

    iid = True
    freeze_support()
    # NAS params
    total_clients = 100
    no_of_clients = 3
    total_search_rounds = 5

    # HPO params
    lr_left = 0.0007
    minibatch_left = 1
    minibatch_right = 256
    no_of_explorations = 10
    explor_epochs = 20

    best_configurations = []
    best_losses = []
    best_models = []

    data_train, data_test = load_dataset()
    if iid:
        data_dict = iid_partition(data_train, total_clients)
    else:
        print('non iid setting ')
        with open('data_division/CIFAR10_dirichelet_08.pkl', 'rb') as file:
            data_dict = pickle.load(file)

    code_strings, messages = initialize_configs(no_of_clients)
    print('number of configs ' + str(len(code_strings)))
    while len(code_strings) != no_of_clients:
        print('not enough configs ')
        code_strings, messages = initialize_configs(no_of_clients)

    print('number of configs ' + str(len(code_strings)))
    best_loss = 0
    best_model = None
    for curr_round in range(total_search_rounds):
        print('Starting search round' + str(curr_round) + '..... ')
        if curr_round != 0:
            code_strings, messages = intermediate_configs(messages, best_loss, best_model, no_of_clients)

        best_config, idx, best_loss, messages, best_model_file = get_best_config(no_of_clients, code_strings,
                                                                                 curr_round, messages, explor_epochs,
                                                                                 data_dict, data_train, data_test)
        if best_config is not None:
            best_configurations.append({'model': str(code_strings[idx]), 'config': str(best_config)})
            best_losses.append(best_loss)
            best_models.append(best_model_file)
            print('round ' + str(curr_round) + 'best config ' + str(best_config))
            print('round ' + str(curr_round) + 'best model ' + str(code_strings[idx]))
            best_model = str(code_strings[idx])

    index_acc_max = np.argmax(np.array(best_losses))
    final_config = best_configurations[index_acc_max]
    with open('final_model_cfa1001.py', 'w') as f:
        f.write(final_config['model'])
        print('final file written')

    with open('final_loss_cfa1001.txt', 'w') as f:
        f.write(str(best_losses[index_acc_max]))
        f.write(str(best_configurations[index_acc_max]))

