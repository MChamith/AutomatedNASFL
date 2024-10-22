import importlib
import inspect
import os
import pickle
import random
import sys
from pathlib import Path
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from non_iid_partition import partition_noniid
from datasets import load_dataset


class CustomDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # print('type ' + str(type(self.idxs[item])))
        data = self.dataset[int(self.idxs[item])]
        image = np.array(data['image'].convert('RGB'))
        label = data['label']
        # print('image ' + str(image))
        # print('label ' + str(label))
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        image = image.permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.int64)

        # print('image shape ' + str(image.shape))

        return image, label

# class CustomDataset(Dataset):
#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = list(idxs)
#
#     def __len__(self):
#         return len(self.idxs)
#
#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         # print('image shape ' + str(image.shape))
#         return image, label

class ClientUpdate(object):
    def __init__(self, dataset, batchSize, learning_rate, epochs, idxs):
        self.train_loader = DataLoader(CustomDataset(dataset, idxs), batch_size=batchSize, shuffle=True)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, model):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.5)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        e_loss = []
        for epoch in range(1, self.epochs + 1):
            train_loss = 0
            model.train()
            for data, labels in self.train_loader:
                data = data.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)

            train_loss = train_loss / len(self.train_loader.dataset)
            e_loss.append(train_loss)

        total_loss = sum(e_loss) / len(e_loss)

        return model.state_dict(), total_loss


def testing(model, dataset, bs, criterion):
    test_loss = 0
    correct = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print('text indexes ' + str([i for i in range(len(dataset))]))
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


def load_dataset_cfar():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    data_train = datasets.CIFAR10(root='./data/cfar', train=True,
                                  download=True, transform=transform)
    data_test = datasets.CIFAR10(root='./data/cfar', train=False,
                                 download=True, transform=transform)

    return data_train, data_test


def convert_data(dataset):
    images = np.array([item['image'].convert('RGB') for item in dataset])
    labels = np.array([item['label'] for item in dataset])
    images, labels = shuffle(images, labels, random_state=0)
    images = np.array([np.array(img) for img in images])
    # combined_dataset = [
    #     (torch.tensor(images, dtype=torch.float32) / 255.0, torch.tensor(labels[i], dtype=torch.int64)) for i in
    #     range(len(images))]
    combined_dataset = [images, labels]
    return combined_dataset


def load_dataset_imgnet():
    print('loading imagenet')
    data_train = load_dataset('Maysee/tiny-imagenet', split='train')
    data_test = load_dataset('Maysee/tiny-imagenet', split='valid')
    print('data train ' + str(data_train[0]))
    # data_train = convert_data(data_train)
    # data_test = convert_data(data_test)

    return data_train, data_test


def iid_partition(dataset, K):
    num_items_per_client = int(len(dataset) / K)
    client_dict = {}
    image_idxs = [i for i in range(len(dataset))]

    for i in range(K):
        client_dict[i] = set(np.random.choice(image_idxs, num_items_per_client, replace=False))
        image_idxs = list(set(image_idxs) - client_dict[i])

    return client_dict


def fedavg_test(model_file, learning_rate):
    global global_model
    iid = True
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device ' + str(device))

    # model_file = "./ModelData2/" + str(nas_round) + '/' + str(client_num) + '/Model' + str(try_no) + '.py'

    path_pyfile = Path(model_file)
    sys.path.append(str(path_pyfile.parent))
    mod_path = str(path_pyfile).replace(os.sep, '.').strip('.py')
    imp_path = importlib.import_module(mod_path)

    for name_local in dir(imp_path):

        if inspect.isclass(getattr(imp_path, name_local)):
            modelClass = getattr(imp_path, name_local)
            global_model = modelClass()
            global_model.to(device)

    global_weights = global_model.state_dict()
    train_loss = []
    test_loss = []
    test_accuracy = []
    T = 100
    C = 0.7
    K = 3
    E = 5
    eta = learning_rate
    B = 64
    B_test = 256
    data_train, data_test = load_dataset_imgnet()
    if iid:
        print('iid setting')
        data_dict = iid_partition(data_train, K)
    else:
        print('non iid setting ')
        with open('data_division/CIFAR10_dirichelet_08.pkl', 'rb') as file:
            data_dict = pickle.load(file)
        # data_dict, num_items, variance = partition_noniid(data_train, 100, 100, 0, True)
    for curr_round in tqdm(range(1, T + 1)):
        # print('Round ',curr_round)
        w, local_loss = [], []
        m = max(int(C * K), 1)

        S_t = np.random.choice(range(K), m, replace=False)
        print('st ' + str(S_t))
        for k in S_t:
            local_update = ClientUpdate(dataset=data_train, batchSize=B, learning_rate=eta, epochs=E, idxs=data_dict[k])
            weights, loss = local_update.train(model=copy.deepcopy(global_model))
            w.append(copy.deepcopy(weights))
            local_loss.append(copy.deepcopy(loss))

        #     if curr_round %  20 == 0:
        #         eta = eta * 0.65

        weights_avg = copy.deepcopy(w[0])
        for k in weights_avg.keys():
            for i in range(1, len(w)):
                weights_avg[k] += w[i][k]

            weights_avg[k] = torch.div(weights_avg[k], len(w))

        global_weights = weights_avg

        # eta = 0.99 * eta
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_loss) / len(local_loss)
        train_loss.append(loss_avg)

        g_loss, g_accuracy = testing(global_model, data_test, B_test, criterion)
        test_loss.append(g_loss)
        test_accuracy.append(g_accuracy)
        print('accuracy ' + str(g_accuracy))

    return test_loss, test_accuracy
