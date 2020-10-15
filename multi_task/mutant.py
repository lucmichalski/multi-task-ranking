
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler

import random
import os
import json


class MUTANT(nn.Module):

    def __init__(self, d_model=768, seq_len=16, dropout=0.1):
        super(MUTANT, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.token_type_embeddings = nn.Embedding(3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Scoreing heads
        self.head_passage = nn.Linear(d_model, 1)
        self.head_entity = nn.Linear(d_model, 1)

    def forward(self, input_CLSs, type_mask):
        # input_CLSs -> [seq_len, batch_size, d_model]
        # type_mask -> [seq_len, batch_size] 0 or 1 for different types

        token_type_embeddings = self.token_type_embeddings(type_mask)

        input_CLSs = input_CLSs + token_type_embeddings

        # Build padding masks i.e. type_mask == 0.
        # src_key_padding_mask = (type_mask > 0).type(torch.int).T
        src_key_padding_mask = (type_mask == 0).T

        # Forward pass of Transformer encoder.
        output_CLSs = self.transformer_encoder(input_CLSs, src_key_padding_mask=src_key_padding_mask)

        # Ensure Passage and Entity heads score correct mask type i.e. passage == 1 & entity = 2.
        passage_mask = (type_mask == 1).type(torch.int).unsqueeze(-1)
        entity_mask = (type_mask == 2).type(torch.int).unsqueeze(-1)
        entity_output = self.head_entity(output_CLSs) * entity_mask
        passage_output = self.head_passage(output_CLSs) * passage_mask

        output = passage_output + entity_output

        return output

    def get_device(self):
        return next(self.parameters()).device

# dir_path='/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/mutant_data/train/'
# doc_to_entity_map_path ='/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/doc_to_entity_map.json'
def get_dev_dataset(save_path_dataset, save_path_dict, dir_path, doc_to_entity_map_path, file_name='_mutant_max.json', max_seq_len=16):
    """ """

    bag_of_CLS = []
    labels = []
    type_mask = []

    dev_run_data_dict = {}

    with open(doc_to_entity_map_path, 'r') as f:
        doc_to_entity_map = json.load(f)

    for path in [dir_path + f for f in os.listdir(dir_path) if file_name in f]:
        with open(path, 'r') as f:
            d = json.load(f)

        query = d['query']['query_id']
        dev_run_data_dict[query] = {}
        results_idx = 0
        for passage_id in d['query']['passage'].keys():
            seq_cls = []
            seq_labels = []
            seq_mask = []

            passage_cls = d['query']['passage'][passage_id]['cls_token']
            passage_relevant = d['query']['passage'][passage_id]['relevant']
            seq_cls.append(passage_cls)
            seq_labels.append([passage_relevant])
            seq_mask.append(1)

            dev_run_data_dict[query][results_idx] = {'doc_id': passage_id, 'relevant': passage_relevant}
            results_idx += 1

            if passage_id in doc_to_entity_map:
                entity_id_list = doc_to_entity_map[passage_id]
                entity_id_list_sorted = [elem for count, elem in
                                         sorted(((entity_id_list.count(e), e) for e in set(entity_id_list)),
                                                reverse=True)]
                for entity_id in entity_id_list_sorted:
                    if len(seq_mask) < max_seq_len:
                        entity_cls = d['query']['passage'][passage_id]['entity'][entity_id]['cls_token']
                        entity_relevant = d['query']['passage'][passage_id]['entity'][entity_id]['relevant']
                        seq_cls.append(entity_cls)
                        seq_labels.append([entity_relevant])
                        seq_mask.append(2)

                        dev_run_data_dict[query][results_idx] = {'doc_id': entity_id, 'relevant': entity_relevant}
                        results_idx += 1

            else:
                # print('{} not in doc_to_entity_map'.format(passage_id))
                for entity_id in d['query']['passage'][passage_id]['entity']:
                    if len(seq_mask) < max_seq_len:
                        entity_cls = d['query']['passage'][passage_id]['entity'][entity_id]['cls_token']
                        entity_relevant = d['query']['passage'][passage_id]['entity'][entity_id]['relevant']
                        seq_cls.append(entity_cls)
                        seq_labels.append([entity_relevant])
                        seq_mask.append(2)

                        dev_run_data_dict[query][results_idx] = {'doc_id': entity_id, 'relevant': entity_relevant}
                        results_idx += 1

            if len(seq_mask) < max_seq_len:
                padding_len = max_seq_len - len(seq_mask)
                for i in range(padding_len):
                    seq_cls.append([0] * 768)
                    seq_labels.append([0])
                    seq_mask.append(0)

            bag_of_CLS.append(seq_cls)
            labels.append(seq_labels)
            type_mask.append(seq_mask)

    bag_of_CLS_tensor = torch.tensor(bag_of_CLS)
    type_mask_tensor = torch.tensor(type_mask)
    labels_tensor = torch.tensor(labels)
    print(bag_of_CLS_tensor.shape, type_mask_tensor.shape, labels_tensor.shape)

    dev_dataset = TensorDataset(bag_of_CLS_tensor, type_mask_tensor, labels_tensor)

    torch.save(obj=dev_dataset, f=save_path_dataset)

    with open(save_path_dict, 'w') as f:
        json.dump(dev_run_data_dict, f)


# dir_path='/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/mutant_data/train/'
# doc_to_entity_map_path ='/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/doc_to_entity_map.json'
def get_train_dataset(save_path_dataset, dir_path, doc_to_entity_map_path, file_name='_mutant_max.json', max_seq_len=16):
    """ """

    bag_of_CLS_R = []
    labels_R = []
    type_mask_R = []
    bag_of_CLS_N = []
    labels_N = []
    type_mask_N = []

    with open(doc_to_entity_map_path, 'r') as f:
        doc_to_entity_map = json.load(f)

    for path in [dir_path + f for f in os.listdir(dir_path) if file_name in f]:
        with open(path, 'r') as f:
            d = json.load(f)
        for passage_id in d['query']['passage'].keys():
            seq_cls = []
            seq_labels = []
            seq_mask = []

            passage_cls = d['query']['passage'][passage_id]['cls_token']
            passage_relevant = d['query']['passage'][passage_id]['relevant']
            seq_cls.append(passage_cls)
            seq_labels.append([passage_relevant])
            seq_mask.append(1)

            if passage_id in doc_to_entity_map:
                entity_id_list = doc_to_entity_map[passage_id]
                entity_id_list_sorted = [elem for count, elem in
                                         sorted(((entity_id_list.count(e), e) for e in set(entity_id_list)),
                                                reverse=True)]
                for entity_id in entity_id_list_sorted:
                    if len(seq_mask) < max_seq_len:
                        entity_cls = d['query']['passage'][passage_id]['entity'][entity_id]['cls_token']
                        entity_relevant = d['query']['passage'][passage_id]['entity'][entity_id]['relevant']
                        seq_cls.append(entity_cls)
                        seq_labels.append([entity_relevant])
                        seq_mask.append(2)

            else:
                # print('{} not in doc_to_entity_map'.format(passage_id))
                for entity_id in d['query']['passage'][passage_id]['entity']:
                    if len(seq_mask) < max_seq_len:
                        entity_cls = d['query']['passage'][passage_id]['entity'][entity_id]['cls_token']
                        entity_relevant = d['query']['passage'][passage_id]['entity'][entity_id]['relevant']
                        seq_cls.append(entity_cls)
                        seq_labels.append([entity_relevant])
                        seq_mask.append(2)

            if len(seq_mask) < max_seq_len:
                padding_len = max_seq_len - len(seq_mask)
                for i in range(padding_len):
                    seq_cls.append([0] * 768)
                    seq_labels.append([0])
                    seq_mask.append(0)

            if passage_relevant == 0:
                bag_of_CLS_N.append(seq_cls)
                labels_N.append(seq_labels)
                type_mask_N.append(seq_mask)
            else:
                bag_of_CLS_R.append(seq_cls)
                labels_R.append(seq_labels)
                type_mask_R.append(seq_mask)

    print('-> {} training R examples'.format(len(bag_of_CLS_R)))
    print('-> {} training N examples'.format(len(bag_of_CLS_N)))
    idx_list = list(range(len(bag_of_CLS_R)))
    diff = len(bag_of_CLS_N) - len(bag_of_CLS_R)
    # randomly sample diff number of samples.
    for idx in random.choices(idx_list, k=diff):
        bag_of_CLS_N.append(bag_of_CLS_R[idx])
        labels_N.append(labels_R[idx])
        type_mask_N.append(type_mask_R[idx])

    bag_of_CLS_tensor = torch.tensor(bag_of_CLS_N)
    type_mask_tensor = torch.tensor(type_mask_N)
    labels_tensor = torch.tensor(labels_N)
    print(bag_of_CLS_tensor.shape, type_mask_tensor.shape, labels_tensor.shape)

    train_dataset = TensorDataset(bag_of_CLS_tensor, type_mask_tensor, labels_tensor)

    torch.save(obj=train_dataset, f=save_path_dataset)


def train_and_dev_mutant(dev_save_path_dict, dev_save_path_dataset, train_save_path_dataset, lr=0.0001, epoch=5, max_seq_len=16, batch_size=32):
    """"""
    print('BUILDING TRAINING DATASET')
    train_dataset = torch.load(train_save_path_dataset)
    print('BUILDING DEV DATASET')
    dev_dataset = torch.load(dev_save_path_dataset)

    # with open(dev_save_path_dict, 'r') as f:
    #     dev_run_dict = json.load(f)

    train_data_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    dev_data_loader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=batch_size)

    model = MUTANT(d_model=768, seq_len=max_seq_len, dropout=0.1)

    # Use GPUs if available.
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU: {}'.format(torch.cuda.get_device_name(0)))
        model.cuda()
        device = torch.device("cuda")

    # Otherwise use CPU.
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    print('RUN EXPERIMENT')
    for i in range(epoch):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
        train_loss_total = 0.0

        model.train()
        for i_train, train_batch in enumerate(train_data_loader):

            bag_of_CLS, type_mask, labels = train_batch
            bag_of_CLS = bag_of_CLS.permute(1, 0, 2)
            type_mask = type_mask.permute(1, 0)
            labels = labels.permute(1, 0, 2)

            model.zero_grad()

            outputs = model.forward(bag_of_CLS.to(device), type_mask=type_mask.to(device))

            loss = loss_func(outputs.cpu(), labels)
            loss.sum().backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            train_loss_total += loss.sum().item()

            if i_train % 1000 == 0:
                print('--------')
                print('train loss @ step {}, {}'.format(i_train, train_loss_total / (i_train + 1)))

                dev_loss_total = 0
                for dev_batch in dev_data_loader:
                    bag_of_CLS, type_mask, labels = dev_batch
                    bag_of_CLS = bag_of_CLS.permute(1, 0, 2)
                    type_mask = type_mask.permute(1, 0)
                    labels = labels.permute(1, 0, 2).type(torch.float)

                    with torch.no_grad():
                        outputs = model.forward(bag_of_CLS.to(device), type_mask=type_mask.to(device))

                    loss = loss_func(outputs.cpu(), labels)

                    dev_loss_total += loss.sum().item()

                print('dev loss @ step {}, {}'.format(i_train, dev_loss_total / (len(dev_data_loader) + 1)))
