
from torch.utils.data import Dataset, DataLoader

import re
import torch
import os


class BertDataset(Dataset):
    """ PyTorch subclass that loads a folder of individual datasets into a single iterable object. """
    def __init__(self, data_dir_path):
        # Store dataset.
        self.samples = []

        # Unpack dataset in sorted order based on file name.
        path_list = os.listdir(data_dir_path)
        print(path_list)
        path_list.sort(key=lambda f: int(float(re.sub('\D', '', f))))
        print('ordered files: {}'.format(path_list))
        for file_name in path_list:
            if file_name[len(file_name)-3:] == '.pt':
                path = os.path.join(data_dir_path, file_name)
                print('loadings data from: {}'.format(path))
                dataset = torch.load(path)
                for td in dataset:
                    self.samples.append(td)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



if __name__ == '__main__':
    data_dir_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'results')

    test_dataset = BertDataset(data_dir_path=data_dir_path)
    print(len(test_dataset))
    dataloader = DataLoader(test_dataset, batch_size=12, shuffle=False)
    for i, batch in enumerate(dataloader):
        print('==================================')
        print(i)
        print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape)
