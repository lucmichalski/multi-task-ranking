
from torch.utils.data import Dataset, DataLoader

import torch
import os


class BertDataset(Dataset):
    def __init__(self, data_dir_path):
        self.samples = []

        for file_name in sorted(os.listdir(data_dir_path)):
            path = os.path.join(data_dir_path, file_name)
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
    dataloader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=1)
    for i, batch in enumerate(dataloader):
        print('==================================')
        print(i)
        for b in batch:
            print(b)
