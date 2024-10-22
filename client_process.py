import os
import random
import time

import GPUtil
import psutil
import torch
import torch.nn as nn

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
import numpy as np


def testing(model, dataset, bs, criterion):
    test_loss = 0
    correct = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_loader = DataLoader(CustomDataset(dataset, [i for i in range(len(dataset))]), batch_size=bs)
    l = len(test_loader)
    model.eval()
    for data, labels in test_loader:
        data = data.to(device)
        labels = labels.to(device)
        output = model(data)
        loss = criterion(output, labels)
        test_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct += pred.eq(labels.data.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, test_accuracy


def client_run_hpo(configs, eta, r, client_num, code_string, nas_round, hpo_round, try_no, explor_epochs, data_dict,
                   train_data, test_data):
    print('starting nas round = ' + str(nas_round) + ' client number= ' + str(client_num) + ' hpo round ' + str(
        hpo_round))
    # X_train = np.load('cfar10_50/X' + str(client_num) + '.npy')
    # y_train = np.load('cfar10_50/y' + str(client_num) + '.npy')
    idxs = data_dict[client_num]
    # print('idxs ' + str(idxs))
    # print('idxs type ' + str(type(idxs)))
    filename = "./ModelDatatinyImagenetNew1/" + str(nas_round) + '/' + str(client_num) + '/Model' + str(
        hpo_round) + '_' + str(
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

        n_i = int(np.floor(n * 0.5))  # no. of configs to keep after pruning
        print('config length ' + str(n))
        # print('number of configs after pruning ' + str(n_i))
        # print('number of configs ' +str(n))
        # print('train size ' + str(r_i))
        idx_subset = random.sample(list(idxs), int(r_i))
        # splits = StratifiedShuffleSplit(n_splits=1, test_size=None, train_size=int(r_i))
        # for train_index, _ in splits.split(X_train, y_train):
        #     X_subset = X_train[train_index]
        #     y_subset = y_train[train_index]
        print('num of datapoints ' + str(len(idx_subset)))
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

            train_loader = DataLoader(CustomDataset(train_data, idx_subset), batch_size=mini_batch, shuffle=True)
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
            g_loss, g_accuracy = testing(model, test_data, 128, criterion)

            performance[str(config)] = g_accuracy
            accuracy_per[str(config)] = g_accuracy

        # prune after sort
        sorted_configs = sorted(configs, key=lambda x: -performance[str(x)])
        configs = sorted_configs[:n_i]
        r_i = max(min(2 * r_i, len(idxs) - 10), 40)
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


def process_client(cl_num, code_strings, curr_round, explor_epochs, messages, data_dict, train_data, test_data):
    print('processing client ' + str(cl_num))
    configurations = []
    accs = []
    model_files = []
    lr_left = 0.0007
    start_time = time.time()
    process = psutil.Process(os.getpid())

    # Initial memory usage
    initial_memory = process.memory_info().rss  # in bytes
    initial_cpu_times = process.cpu_times()

    # Get GPU stats before computation
    gpus_before = GPUtil.getGPUs()

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
                best_config, final_best_acc, final_best_model_file = client_run_hpo(configs, 2, 256, cl_num,
                                                                                    code_strings[cl_num], curr_round,
                                                                                    hpo_round, try_no, explor_epochs,
                                                                                    data_dict, train_data, test_data)
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
        best_acc = 0
        best_model_file = None

    # Get GPU stats after computation
    gpus_after = GPUtil.getGPUs()

    # Calculate elapsed time and final memory usage
    elapsed_time = time.time() - start_time
    final_memory = process.memory_info().rss  # in bytes
    final_cpu_times = process.cpu_times()

    # Calculate memory and CPU usage statistics
    memory_usage = (final_memory - initial_memory) / (1024 * 1024)  # Convert bytes to MB
    cpu_usage = final_cpu_times.user - initial_cpu_times.user  # User CPU time difference

    # Get GPU usage (assuming each client uses only one GPU)
    gpu_usage_info = []
    # for gpu_before, gpu_after in zip(gpus_before, gpus_after):
    #     gpu_usage_info.append({
    #         'GPU ID': gpu_before.id,
    #         'Memory Used (MB)': gpu_after.memoryUsed - gpu_before.memoryUsed,
    #         'Load (%)': (gpu_after.load - gpu_before.load) * 100,
    #         'Temperature (°C)': gpu_after.temperature
    #     })
    #
    # # Log the resource usage
    # print(f"Client {cl_num}: CPU Time: {cpu_usage} seconds, Memory Usage: {memory_usage:.2f} MB")
    # for gpu_info in gpu_usage_info:
    #     with open('cfar100_compute_2.txt', 'w') as fw:
    #         fw.write(
    #             f"Client {cl_num}: GPU ID: {gpu_info['GPU ID']}, GPU Memory Used: {gpu_info['Memory Used (MB)']:.2f} MB, GPU Load: {gpu_info['Load (%)']:.2f}%, GPU Temperature: {gpu_info['Temperature (°C)']} °C\n")
    #         fw.write("Num data points " + str(len(data_dict[0])))
    return best_config, best_acc, best_model_file


def get_best_config(no_of_clients, code_strings, curr_round, messages, explor_epochs, data_dict, data_train, data_test):
    # client_ids = np.random.choice(range(100), 10, replace=False)
    client_ids = [i for i in range(3)]
    code_strings_dict = {}
    for i in range(len(client_ids)):
        code_strings_dict[client_ids[i]] = code_strings[i]
    with multiprocessing.Pool(processes=no_of_clients) as pool:
        process_client_partial = partial(process_client, code_strings=code_strings_dict, curr_round=curr_round,
                                         explor_epochs=explor_epochs, messages=messages, data_dict=data_dict,
                                         train_data=data_train, test_data=data_test)
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
        with open('best_model_tinyimagenet_new1/round_' + str(curr_round) + '.txt', 'w') as fw:
            fw.write(str(best_model_file))
            fw.write(str(final_config))
        return final_config, index_max, best_loss, messages, best_model_file

    else:
        return None, None, None, None, None
