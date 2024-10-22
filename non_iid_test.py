from torchvision import datasets, transforms
from non_iid_partition import partition_noniid
from non_iid_dirichlet import non_iid_dirichlet_split

import matplotlib.pyplot as plt

def load_dataset():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    data_train = datasets.CIFAR10(root='./data/cfar', train=True,
                                        download=True, transform=transform)
    data_test = datasets.CIFAR10(root='./data/cfar', train=False,
                                       download=True, transform=transform)

    return data_train, data_test


train_data , test_data = load_dataset()

dict_users, num_items, variance = partition_noniid(train_data, 100, 100, 0, True)
# client_data_indices, client_class_counts = non_iid_dirichlet_split(train_data, 100, 0.01, 10)
print(dict_users)
print(type(print(dict_users)))
# print('num items ' + str(num_items))

