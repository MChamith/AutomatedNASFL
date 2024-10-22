import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
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
