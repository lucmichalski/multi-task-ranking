
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from learning.utils import MutantDataset
from retrieval.tools import SearchTools, EvalTools

import random
import itertools
import os
import json
import re

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
def get_dev_dataset(save_path_dataset, save_path_run, dir_path, doc_to_entity_map_path, file_name='_mutant_max.json', max_seq_len=16):
    """ """
    bag_of_CLS = []
    labels = []
    type_mask = []

    dev_run_data = []

    with open(doc_to_entity_map_path, 'r') as f:
        doc_to_entity_map = json.load(f)

    for path in [dir_path + f for f in os.listdir(dir_path) if file_name in f]:
        with open(path, 'r') as f:
            d = json.load(f)

        for passage_id in d['query']['passage'].keys():
            seq_cls = []
            seq_labels = []
            seq_mask = []
            seq_run = []

            query = d['query']['query_id']
            seq_run.append(query)

            passage_cls = d['query']['passage'][passage_id]['cls_token']
            passage_relevant = float(d['query']['passage'][passage_id]['relevant'])
            seq_cls.append(passage_cls)
            seq_labels.append([passage_relevant])
            seq_mask.append(1)

            seq_run.append(passage_id)
            seq_run.append(passage_relevant)

            if passage_id in doc_to_entity_map:
                entity_id_list = doc_to_entity_map[passage_id]
                entity_id_list_sorted = [elem for count, elem in sorted(((entity_id_list.count(e), e) for e in set(entity_id_list)), reverse=True)]
                for entity_id in entity_id_list_sorted:
                    if len(seq_mask) < max_seq_len:
                        entity_cls = d['query']['passage'][passage_id]['entity'][entity_id]['cls_token']
                        entity_relevant = float(d['query']['passage'][passage_id]['entity'][entity_id]['relevant'])
                        seq_cls.append(entity_cls)
                        seq_labels.append([entity_relevant])
                        seq_mask.append(2)

                        seq_run.append(entity_id)
                        seq_run.append(entity_relevant)


            else:
                # print('{} not in doc_to_entity_map'.format(passage_id))
                for entity_id in d['query']['passage'][passage_id]['entity']:
                    if len(seq_mask) < max_seq_len:
                        entity_cls = d['query']['passage'][passage_id]['entity'][entity_id]['cls_token']
                        entity_relevant = float(d['query']['passage'][passage_id]['entity'][entity_id]['relevant'])
                        seq_cls.append(entity_cls)
                        seq_labels.append([entity_relevant])
                        seq_mask.append(2)

                        seq_run.append(entity_id)
                        seq_run.append(entity_relevant)

            if len(seq_mask) < max_seq_len:
                padding_len = max_seq_len - len(seq_mask)
                for i in range(padding_len):
                    seq_cls.append([0.0] * 768)
                    seq_labels.append([0.0])
                    seq_mask.append(0)

                    seq_run.append("PAD")
                    seq_run.append("PAD")

            bag_of_CLS.append(seq_cls)
            labels.append(seq_labels)
            type_mask.append(seq_mask)
            dev_run_data.append(seq_run)

    bag_of_CLS_tensor = torch.tensor(bag_of_CLS)
    type_mask_tensor = torch.tensor(type_mask)
    labels_tensor = torch.tensor(labels)
    print(bag_of_CLS_tensor.shape, type_mask_tensor.shape, labels_tensor.shape)

    dev_dataset = TensorDataset(bag_of_CLS_tensor, type_mask_tensor, labels_tensor)

    torch.save(obj=dev_dataset, f=save_path_dataset)

    with open(save_path_run, 'w') as f:
        for data in dev_run_data:
            f.write(" ".join((str(i) for i in data)) + '\n')


# dir_path='/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/mutant_data/train/'
# doc_to_entity_map_path ='/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/doc_to_entity_map.json'
def get_train_dataset(save_path_dataset_dir, dir_paths, doc_to_entity_map_path, file_name='_mutant_max.json', max_seq_len=16,
                      max_rank=100):
    """ """

    bag_of_CLS_R = []
    labels_R = []
    type_mask_R = []
    bag_of_CLS_N = []
    labels_N = []
    type_mask_N = []
    chunk_i = 0
    counter = 0

    if not os.path.exists(save_path_dataset_dir):
        os.mkdir(save_path_dataset_dir)

    with open(doc_to_entity_map_path, 'r') as f:
        doc_to_entity_map = json.load(f)

    for dir_path in dir_paths:
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

            counter += 1
            if counter % 100 == 0:
                # write_batch
                chunk_path = save_path_dataset_dir + 'chunk_{}.pt'.format(chunk_i)
                print('------- CHUNK {} -------'.format(chunk_i))
                print(chunk_path)
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

                torch.save(obj=train_dataset, f=chunk_path)

                bag_of_CLS_R = []
                labels_R = []
                type_mask_R = []
                bag_of_CLS_N = []
                labels_N = []
                type_mask_N = []
                chunk_i += 1

    if bag_of_CLS_R > 0:
        chunk_path = save_path_dataset_dir + 'chunk_{}.pt'.format(chunk_i)
        print('------- CHUNK {} -------'.format(chunk_i))
        print(chunk_path)
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

        torch.save(obj=train_dataset, f=chunk_path)

        bag_of_CLS_R = []
        labels_R = []
        type_mask_R = []
        bag_of_CLS_N = []
        labels_N = []
        type_mask_N = []
        chunk_i += 1






def unpack_run_data(run_data, max_seq_len=16):
    """"""
    data = []
    for run in run_data:
        query = run[0]
        results = run[1:]
        loops = int(max_seq_len)
        start_i = 0
        end_i = 1
        task = 'doc'
        for i in range(loops):
            if results[start_i] == 'PAD':
                task = 'PAD'
            data.append((task, query, results[start_i], results[end_i]))
            start_i += 2
            end_i += 2
            task = 'entity'
    return data


def update_topic_passage_run(topic_query, topic_run_passage_dict, dev_qrels, max_rank):
    """"""
    if topic_query in dev_qrels:
        R = len(dev_qrels[topic_query])
    else:
        R = 0

    topic_run_data = [v for k, v in sorted(topic_run_passage_dict.items(), key=lambda item: item[1][1], reverse=True)][:max_rank]
    assert len(topic_run_data) <= max_rank, topic_run_data
    topic_run_data.sort(key=lambda x: x[1], reverse=True)
    topic_run = [i[0] for i in topic_run_data]
    map = EvalTools().get_map(run=topic_run, R=R)
    return map


def update_topic_entity_run(topic_query, topic_run_entity_dict, dev_qrels, max_rank):
    """"""
    if topic_query in dev_qrels:
        R = len(dev_qrels[topic_query])
    else:
        R = 0

    topic_run_data = [v for k, v in sorted(topic_run_entity_dict.items(), key=lambda item: item[1][1], reverse=True)][:max_rank]
    assert len(topic_run_data) <= max_rank, topic_run_data
    topic_run_data.sort(key=lambda x: x[1], reverse=True)
    topic_run = [i[0] for i in topic_run_data]
    map = EvalTools().get_map(run=topic_run, R=R)
    return map


def get_loss(outputs, labels, max_seq_len):
    """"""
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    passage_outputs = outputs[:,0,:]
    passage_labels = labels[:,0,:]
    passage_loss = loss_func(passage_outputs, passage_labels) * max_seq_len

    entity_outputs = outputs[:,1:,:]
    entity_labels = labels[:,1:,:]
    entity_loss = loss_func(entity_outputs, entity_labels)

    return passage_loss.sum() + entity_loss.sum()


def run_validation(model, dev_data_loader, dev_run_data, max_seq_len, device, i_train, dev_qrels, max_rank):

    dev_labels = []
    dev_scores = []
    dev_loss_total = 0
    for dev_batch in dev_data_loader:
        bag_of_CLS, type_mask, labels = dev_batch
        bag_of_CLS = bag_of_CLS.permute(1, 0, 2)
        type_mask = type_mask.permute(1, 0)
        labels = labels.permute(1, 0, 2).type(torch.float)

        with torch.no_grad():
            outputs = model.forward(bag_of_CLS.to(device), type_mask=type_mask.to(device))

        loss = get_loss(outputs.cpu(), labels, max_seq_len)

        dev_loss_total += loss.item()
        dev_scores += list(itertools.chain(*outputs.permute(1, 0, 2).cpu().numpy().tolist()))
        dev_labels += list(itertools.chain(*labels.permute(1, 0, 2).cpu().numpy().tolist()))

    unpacked_dev_run_data = unpack_run_data(dev_run_data, max_seq_len=max_seq_len)

    assert len(dev_scores) == len(dev_labels) == len(dev_run_data)*max_seq_len == len(unpacked_dev_run_data), '{} == {} == {} == {}'.format(len(dev_scores), len(dev_labels), len(dev_run_data)*max_seq_len, len(unpacked_dev_run_data))
    topic_query_passage = None
    topic_query_entity = None
    # original_map_sum = 0.0
    map_sum_passage = 0.0
    map_sum_entity = 0.0
    topic_counter_passage = 0
    topic_counter_entity = 0
    passage_score = 0.0
    topic_run_passage_dict = {}
    topic_run_entity_dict = {}
    print('dev loss @ step {}, {}'.format(i_train, dev_loss_total / (len(dev_data_loader) + 1)))
    for dev_label, dev_score, unpack_run in zip(dev_labels, dev_scores, unpacked_dev_run_data):

        task, query, doc_id, label = unpack_run[0], unpack_run[1], unpack_run[2], unpack_run[3]
        if 'doc' == task:
            assert float(dev_label[0]) == float(label), '{} == {}'.format(dev_label[0], label)

            if (topic_query_passage != None) and (topic_query_passage != query):
                run_map = update_topic_passage_run(topic_query_passage, topic_run_passage_dict, dev_qrels, max_rank)
                map_sum_passage += run_map
                # Start new topic run.
                topic_counter_passage += 1
                topic_run_passage_dict = {}

            topic_run_passage_dict[doc_id] = [float(label), float(dev_score[0])]

            # Update topic run.
            topic_query_passage = query
            passage_score = float(dev_score[0])

        if 'entity' == task:
            assert float(dev_label[0]) == float(label), '{} == {}'.format(dev_label[0], label)

            if (topic_query_entity != None) and (topic_query_entity != query):
                run_map = update_topic_entity_run(topic_query_entity, topic_run_entity_dict, dev_qrels, max_rank)
                map_sum_entity += run_map
                # Start new topic run.
                topic_counter_entity += 1
                topic_run_entity_dict = {}

            entity_score = passage_score * float(dev_score[0])
            topic_run_entity_dict[doc_id] = [float(label), entity_score]

            if doc_id in topic_run_entity_dict:
                if entity_score > topic_run_entity_dict[doc_id][1]:
                    topic_run_entity_dict[doc_id] = [float(label), entity_score]
            else:
                topic_run_entity_dict[doc_id] = [float(label), entity_score]

            # Update topic run.
            topic_query_entity = query

    if len(topic_run_passage_dict) > 0:
        run_map = update_topic_passage_run(topic_query_passage, topic_run_passage_dict, dev_qrels, max_rank)
        map_sum_passage += run_map
        # Start new topic run.
        topic_counter_passage += 1

    if len(topic_run_entity_dict) > 0:
        run_map = update_topic_entity_run(topic_query_entity, topic_run_entity_dict, dev_qrels, max_rank)
        map_sum_passage += run_map
        # Start new topic run.
        topic_counter_entity += 1

    assert topic_counter_passage == topic_counter_entity, "{} == {}".format(topic_counter_passage, topic_counter_entity)
    print('Passage MAP: {}'.format(map_sum_passage/topic_counter_passage))
    print('Entity MAP: {}'.format(map_sum_entity/topic_counter_entity))


def train_and_dev_mutant(dev_save_path_run, dev_save_path_dataset, dev_qrels_path, train_save_path_dataset, lr=0.0001, epoch=5,
                         max_seq_len=16, batch_size=32, max_rank=100):
    """"""
    # print('BUILDING TRAINING DATASET')
    # train_dataset = MutantDataset(train_save_path_dataset)
    print('BUILDING DEV DATASET')
    dev_dataset = torch.load(dev_save_path_dataset)

    dev_run_data = []
    with open(dev_save_path_run, 'r') as f:
        for line in f:
            data = line.strip().split()
            dev_run_data.append(data)

    dev_qrels = SearchTools().retrieval_utils.get_qrels_binary_dict(dev_qrels_path)
    # train_data_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
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
        print('\\\\\\\\\\\\\\\\\\\\\\\\\\')
        print('======= EPOCH {} ========'.format(i))
        print('\\\\\\\\\\\\\\\\\\\\\\\\\\')

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_loss_total = 0.0
        i_train = 0

        # Unpack dataset in sorted order based on file name.
        path_list = os.listdir(train_save_path_dataset)
        path_list = [path for path in path_list if '.pt' in path]
        path_list.sort(key=lambda f: int(float(re.sub('\D', '', f))))
        print('ordered files: {}'.format(path_list))
        for file_name in path_list:
            if file_name[len(file_name) - 3:] == '.pt':
                path = os.path.join(train_save_path_dataset, file_name)
                print('loadings data from: {}'.format(path))
                train_dataset = torch.load(path)

            train_data_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

            model.train()
            for train_batch in train_data_loader:
                #############################################
                ################## TRAIN ####################
                #############################################
                bag_of_CLS, type_mask, labels = train_batch
                bag_of_CLS = bag_of_CLS.permute(1, 0, 2).type(torch.float)
                type_mask = type_mask.permute(1, 0)
                labels = labels.permute(1, 0, 2).type(torch.float)

                model.zero_grad()

                outputs = model.forward(bag_of_CLS.to(device), type_mask=type_mask.to(device))

                loss = get_loss(outputs.cpu(), labels, max_seq_len)
                loss.backward()
                optimizer.step()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                train_loss_total += loss.item()
                i_train += 1

                if i_train % 2500 == 0:
                    #############################################
                    ################## VALID ####################
                    #############################################
                    print('--------')
                    print('train loss @ step {}, {}'.format(i_train, train_loss_total / (i_train + 1)))

                    run_validation(model, dev_data_loader, dev_run_data, max_seq_len, device, i_train, dev_qrels,
                                   max_rank)

    run_validation(model, dev_data_loader, dev_run_data, max_seq_len, device, i_train, dev_qrels,
                   max_rank)

    print('*** EPOCH {} *** train loss @ step {}, {}'.format(i, i_train, train_loss_total / (i_train + 1)))

    run_validation(model, dev_data_loader, dev_run_data, max_seq_len, device, i_train, dev_qrels,
                   max_rank)
    print('========================')
