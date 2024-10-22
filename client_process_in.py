import os
import random

import torch
import torch.nn as nn
import numpy as np
from prettytable import PrettyTable
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
import importlib
import inspect
from pathlib import Path
import sys
from data_util import CustomDataset
import multiprocessing
from functools import partial
from random import randrange
from gpt_util import handle_error


def testing(model, dataset, bs, criterion):
    test_loss = 0
    correct = 0
    test_loader = DataLoader(dataset, batch_size=bs)
    l = len(test_loader)
    model.eval()
    for data, labels in test_loader:
        # data, labels = data.cuda(), labels.cuda()
        output = model(data)
        loss = criterion(output, labels)
        test_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct += pred.eq(labels.data.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, test_accuracy



def client_run_hpo(configs, eta, r, client_num, code_string, nas_round, hpo_round, try_no, explor_epochs):
    print('starting nas round = ' + str(nas_round) + ' client number= ' + str(client_num) + ' hpo round ' + str(
        hpo_round))
    X_train = np.load('cfar10_50/X' + str(client_num) + '.npy')
    y_train = np.load('cfar10_50/y' + str(client_num) + '.npy')

    X_test = np.load('cfar10_50/X_test.npy')
    y_test = np.load('cfar10_50/y_test.npy')
    print('test test size ' + str(X_test.shape))
    filename = "./ModelData6/" + str(nas_round) + '/' + str(client_num) + '/Model' + str(hpo_round) + '_' + str(
        try_no) + '.py'

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print('writing new code string\n' + str(code_string))
    with open(filename, 'w') as f:
        f.write(code_string)
        print('file written ' + str(filename))

    ptable = PrettyTable()
    ptable.field_names = ["Num Configs", "Best Config", "Accuracy"]

    final_best_config = None
    final_best_accuracy = 0
    r_i = r
    round_ = 0
    while len(configs) > 1:
        n = len(configs)

        n_i = int(np.floor(n * 0.75))  # no. of configs to keep after pruning
        # print('number of configs after pruning ' + str(n_i))
        # print('number of configs ' +str(n))
        # print('train size ' + str(r_i))
        splits = StratifiedShuffleSplit(n_splits=1, test_size=None, train_size=int(r_i))
        for train_index, _ in splits.split(X_train, y_train):
            X_subset = X_train[train_index]
            y_subset = y_train[train_index]

        performance = {}
        accuracy_per = {}

        for config in configs:

            path_pyfile = Path(filename)
            sys.path.append(str(path_pyfile.parent))
            mod_path = str(path_pyfile).replace(os.sep, '.').strip('.py')
            imp_path = importlib.import_module(mod_path)

            for name_local in dir(imp_path):

                if inspect.isclass(getattr(imp_path, name_local)):
                    modelClass = getattr(imp_path, name_local)
                    model = modelClass()
                    model.to(device)

            mini_batch = 64
            lr = configs[0]['learning_rate']
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            #             if round_ != 0:
            # #                 print('loading state')
            #                 checkpoint = torch.load('ModelData/model' + str(config['config_id'])+'.pt')

            #                 model.load_state_dict(checkpoint['model_state_dict'])
            #                 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            train_loader = DataLoader(CustomDataset(X_subset, y_subset), batch_size=mini_batch, shuffle=True)
            for epoch in range(1, explor_epochs):
                train_loss = 0
                model.train()
                for data, labels in train_loader:
                    # data, labels = data.cuda(), labels.cuda()
                    data = data.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * data.size(0)

            criterion = nn.CrossEntropyLoss()
            g_loss, g_accuracy = testing(model, X_test, y_test, 32, criterion)

            performance[str(config)] = g_accuracy
            accuracy_per[str(config)] = g_accuracy

        # prune after sort
        sorted_configs = sorted(configs, key=lambda x: -performance[str(x)])
        configs = sorted_configs[:n_i]
        r_i = min(2 * r_i, 490)
        # explor_epochs += 5
        best_config_str = max(performance, key=performance.get)
        ptable.add_row([len(configs), best_config_str, performance[best_config_str]])

        if performance[best_config_str] > final_best_accuracy:
            final_best_config = best_config_str
            final_best_loss = performance[best_config_str]
            final_best_accuracy = accuracy_per[best_config_str]
            print('client ' + str(client_num) + ' best accuracy ' + str(final_best_accuracy))
        round_ += 1

    final_ptable = PrettyTable()
    final_ptable.field_names = ["Final Best Config", "Final Best Accuracy"]
    final_ptable.add_row([final_best_config, final_best_loss])
    print("Final Best Configuration:")
    print(final_ptable)
    return final_best_config, final_best_accuracy, filename


def process_client(cl_num, code_strings, curr_round, explor_epochs, messages):
    print('processing client ' + str(cl_num))
    configurations = []
    accs = []
    model_files = []
    lr_left = 0.0007

    for hpo_round in range(2):
        lr_values = []
        abondoning = False
        for i in range(2):
            for j in range(10):
                lr_val = float(
                    randrange(int(lr_left * 1000000 * pow(2, j)), int(lr_left * 1000000 * pow(2, j + 1)))) / 1000000
                lr_values.append(lr_val)

        config_id = 0
        initial_configs = []
        for lr in lr_values:
            initial_configs.append({'config_id': str(config_id), 'learning_rate': lr})
            config_id += 1
        configs = initial_configs.copy()

        test_model = True
        try_no = 0
        while test_model:
            try:
                best_config, final_best_acc, final_best_model_file = client_run_hpo(configs, 2, 128, cl_num,
                                                                                    code_strings[cl_num], curr_round,
                                                                                    hpo_round, try_no, explor_epochs)
                test_model = False
                configurations.append(best_config)
                accs.append(final_best_acc)
                model_files.append(final_best_model_file)
            except Exception as e:
                print(f'Error occurred for client {cl_num}: {str(e)}')
                if try_no > 25:
                    abondoning = True
                    break

                code_str, messages = handle_error(messages, code_strings[cl_num], e)
                if code_str:
                    code_strings[cl_num] = code_str[0]
                try_no += 1

        if abondoning:
            continue
    if len(accs) > 0:
        print('accs exist')
        index_max = np.argmax(np.array(accs))
        best_acc = accs[index_max]
        best_config = configurations[index_max]
        best_model_file = model_files[index_max]
    else:
        print('accs do not exist')
        best_config = None
        best_acc = None
        best_model_file = None
    return best_config, best_acc, best_model_file


def get_best_config(no_of_clients, code_strings, curr_round, messages, explor_epochs):
    client_ids = [random.randint(0, 49) for _ in range(no_of_clients)]
    code_strings_dict = {}
    for i in range(len(client_ids)):
        code_strings_dict[client_ids[i]] = code_strings[i]
    with multiprocessing.Pool(processes=no_of_clients) as pool:
        process_client_partial = partial(process_client, code_strings=code_strings_dict, curr_round=curr_round,
                                         explor_epochs=explor_epochs, messages=messages)
        results = pool.map(process_client_partial, client_ids)

    client_results_configurations, client_results_accs, client_results_models = zip(*results)
    print('client results accs ' + str(client_results_accs))
    if client_results_accs is not None:
        index_max = np.argmax(np.array(client_results_accs))
        print(f'Best result is from client {client_ids[index_max]}')
        final_config = client_results_configurations[index_max]
        print(f'Final best acc: {client_results_accs[index_max]}')
        print(f'Final best config: {final_config}')
        best_loss = client_results_accs[index_max]
        print('final best model ' + str(client_results_models[index_max]))
        best_model_file = client_results_models[index_max]
        with open('best_model6/round_' + str(curr_round) + '.txt', 'w') as fw:
            fw.write(str(best_model_file))
            fw.write(str(final_config))
        return final_config, index_max, best_loss, messages, best_model_file

    else:
        return None, None, None, None, None
