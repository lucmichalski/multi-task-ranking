
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from scipy.spatial import distance

import itertools
import random
import json
import torch
import os

from multi_task.processing import dataset_metadata
from retrieval.tools import EvalTools, SearchTools


def get_dict_from_json(path):
    """"""
    with open(path, 'r') as f:
        d = json.load(f)
    return d


def write_run_to_file(query, run_data, run_path, how):
    """ """
    run_data.sort(key=lambda tup: tup[1], reverse=True)
    rank = 1
    with open(run_path, 'a+') as f:
        for doc_id, score in run_data:
            f.write(' '.join((query, 'Q0', doc_id, str(rank), str(score), how)) + '\n')
            rank += 1


def rerank_runs(dataset,  parent_dir_path='/nfs/trec_car/data/entity_ranking/multi_task_data_by_query/'):
    """ """
    dir_path = parent_dir_path + '{}_data/'.format(dataset)

    passage_qrels = dataset_metadata['passage_' + dataset][1]
    entity_qrels = dataset_metadata['entity_' + dataset][1]

    # entity_links_path = dir_path + 'passage_to_entity.json'
    # entity_links_dict = get_dict_from_json(path=entity_links_path)

    for how in ['euclidean', 'original', 'cosine_sim']:

        for query_path in [dir_path + f for f in os.listdir(dir_path) if 'data.json' in f]:

            # === QUERY DATA ===
            query_dict = get_dict_from_json(path=query_path)
            query = query_dict['query']['query_id']
            query_cls = query_dict['query']['cls_token']

            # === PASSAGE DATA ===
            passage_run_data = []
            for doc_id in query_dict['passage'].keys():
                doc_cls = query_dict['passage'][doc_id]['cls_token']

                if how == 'euclidean':
                    passage_score = - distance.euclidean(query_cls,  doc_cls)
                    passage_run_path = dir_path + how + '_passage.run'
                elif how == 'cosine_sim':
                    passage_score = 1 - distance.cosine(query_cls,  doc_cls)
                    passage_run_path = dir_path + how + '_passage.run'
                elif how == 'original':
                    passage_score = - float(query_dict['passage'][doc_id]['rank'])
                    passage_run_path = dir_path + how + '_passage.run'
                else:
                    raise

                passage_run_data.append((doc_id, passage_score))

            write_run_to_file(query=query, run_data=passage_run_data, run_path=passage_run_path, how=how)

            # === ENTITY DATA ===
            entity_run_data = []
            for doc_id in query_dict['entity'].keys():
                doc_cls = query_dict['entity'][doc_id]['cls_token']

                if how == 'euclidean':
                    entity_score = - distance.euclidean(query_cls, doc_cls)
                    entity_run_path = dir_path + how + '_entity.run'
                elif how == 'cosine_sim':
                    entity_score = 1 - distance.cosine(query_cls, doc_cls)
                    entity_run_path = dir_path + how + '_entity.run'
                elif how == 'original':
                    entity_score = - float(query_dict['entity'][doc_id]['rank'])
                    entity_run_path = dir_path + how + '_entity.run'
                else:
                    raise

                entity_run_data.append((doc_id, entity_score))

            write_run_to_file(query=query, run_data=entity_run_data, run_path=entity_run_path, how=how)

        # === EVAL RUNS ===
        EvalTools().write_eval_from_qrels_and_run(qrels_path=passage_qrels, run_path=passage_run_path)
        EvalTools().write_eval_from_qrels_and_run(qrels_path=entity_qrels, run_path=entity_run_path)


def train_model(batch_size=128, lr=0.0005, parent_dir_path='/nfs/trec_car/data/entity_ranking/multi_task_data_by_query/'):
    """ """
    train_dir_path = parent_dir_path + 'train_data/'
    dev_dir_path = parent_dir_path + 'dev_data/'

    passage_qrels = dataset_metadata['passage_' + 'dev'][1]
    entity_qrels = dataset_metadata['entity_' + 'dev'][1]

    # ==== Build training data ====
    print('Build training data')
    train_input_list_R = []
    train_labels_list_R = []
    train_input_list_N = []
    train_labels_list_N = []
    for train_query_path in [train_dir_path + f for f in os.listdir(train_dir_path) if 'data.json' in f]:

        query_dict = get_dict_from_json(path=train_query_path)
        query = query_dict['query']['query_id']
        query_cls = query_dict['query']['cls_token']

        for doc_id in query_dict['passage'].keys():
            doc_cls = query_dict['passage'][doc_id]['cls_token']
            relevant = float(query_dict['passage'][doc_id]['relevant'])
            input = query_cls + doc_cls
            if relevant == 0:
                train_input_list_N.append(input)
                train_labels_list_N.append([relevant])
            else:
                train_input_list_R.append(input)
                train_labels_list_R.append([relevant])

    print('-> {} training R examples'.format(len(train_input_list_R)))
    print('-> {} training N examples'.format(len(train_input_list_N)))
    idx_list = list(range(len(train_input_list_R)))
    diff = len(train_labels_list_N) - len(train_input_list_R)
    # randomly sample diff number of samples.
    for idx in random.choices(idx_list, k=diff):
        train_input_list_N.append(train_input_list_R[idx])
        train_labels_list_N.append(train_labels_list_R[idx])
    print('-> {} class balancing'.format(len(train_labels_list_N)))
    train_dataset = TensorDataset(torch.tensor(train_input_list_N), torch.tensor(train_labels_list_N))
    train_data_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

    # ==== Build dev data ====

    print('Build dev data')
    dev_input_list = []
    dev_labels_list = []
    dev_run_data = []
    dev_qrels = SearchTools.retrieval_utils.get_qrels_binary_dict(passage_qrels)
    for dev_query_path in [dev_dir_path + f for f in os.listdir(dev_dir_path) if 'data.json' in f]:

        query_dict = get_dict_from_json(path=dev_query_path)
        query = query_dict['query']['query_id']
        query_cls = query_dict['query']['cls_token']

        for doc_id in query_dict['passage'].keys():
            doc_cls = query_dict['passage'][doc_id]['cls_token']
            relevant = float(query_dict['passage'][doc_id]['relevant'])

            input = query_cls + doc_cls
            dev_input_list.append(input)
            dev_labels_list.append([relevant])
            dev_run_data.append([query,doc_id,relevant])

    print('-> {} dev examples'.format(len(dev_labels_list)))
    dev_dataset = TensorDataset(torch.tensor(dev_input_list), torch.tensor(dev_labels_list))
    dev_data_loader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=batch_size)

    # ==== Model setup ====

    model = torch.nn.Sequential(
        torch.nn.Linear(1536, 1536),
        torch.nn.ReLU(),
        torch.nn.Linear(1536, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )

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

    # ==== Experiments ====

    for epoch in range(1,11):

        train_batches = len(train_data_loader)
        dev_batches = len(dev_data_loader)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
        train_loss_total = 0.0
    #
        print('====== EPOCH {} ======'.format(epoch))
        # ========================================
        #               Training
        # ========================================
        for i_train, train_batch in enumerate(train_data_loader):
            model.train()
            model.zero_grad()
            inputs, labels = train_batch
            outputs = model.forward(inputs)

            # Calculate Loss: softmax --> cross entropy loss
            loss = loss_func(outputs, labels)
            # Getting gradients w.r.t. parameters
            loss.sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_total += loss.sum().item()

            if i_train % 500 == 0:

                # ========================================
                #               Validation
                # ========================================

                print('batch: {} / {} -> av. training loss: {}'.format(i_train+1, train_batches, loss/(i_train+1)))
                # print('======= inputs =====')
                # print(inputs)
                # print('======= labels =====')
                # print(labels)
                # print('======= outputs =====')
                # print(outputs)

                model.eval()
                dev_loss_total = 0.0
                dev_score = []
                dev_label = []
                for i_dev, dev_batch in enumerate(dev_data_loader):
                    inputs, labels = dev_batch

                    with torch.no_grad():
                        outputs = model.forward(inputs)
                        loss = loss_func(outputs, labels)

                        dev_loss_total += loss.sum().item()
                        dev_label += list(itertools.chain(*labels.cpu().numpy().tolist()))
                        dev_score += list(itertools.chain(*outputs.cpu().numpy().tolist()))

                assert len(dev_score) == len(dev_label) == len(dev_run_data), "{} == {} == {}".format(len(dev_score), len(dev_label), len(dev_run_data))
                print('av. dev loss = {}'.format(dev_loss_total/(i_dev+1)))

                # Store topic query and count number of topics.
                topic_query = None
                original_map_sum = 0.0
                map_sum = 0.0
                topic_counter = 0
                topic_run_data = []
                for label, score, run_data in zip(dev_label, dev_score, dev_run_data):
                    query, doc_id, label_ground_truth = run_data

                    assert label == label_ground_truth, "score {} == label_ground_truth {}".format(label, label_ground_truth)

                    if (topic_query != None) and (topic_query != query):
                        if topic_query in dev_qrels:
                            R = len(dev_qrels[topic_query])
                        else:
                            R = 0
                        original_run = [i[0] for i in topic_run_data]
                        original_map_sum += EvalTools().get_map(run=original_run, R=R)
                        topic_run_data.sort(key=lambda x: x[1], reverse=True)
                        topic_run = [i[0] for i in topic_run_data]
                        map_sum += EvalTools().get_map(run=topic_run, R=R)
                        # Start new topic run.
                        topic_counter += 1
                        topic_run_data = []

                    topic_run_data.append([label, score])
                    # Update topic run.
                    topic_query = query

                if len(topic_run_data) > 0:
                    if topic_query in dev_qrels:
                        R = len(dev_qrels[topic_query])
                    else:
                        R = 0
                    original_run = [i[0] for i in topic_run_data]
                    original_map_sum += EvalTools().get_map(run=original_run, R=R)
                    topic_run_data.sort(key=lambda x: x[1], reverse=True)
                    topic_run = [i[0] for i in topic_run_data]
                    map_sum += EvalTools().get_map(run=topic_run, R=R)

                print('Original MAP = {}'.format(original_map_sum/topic_counter))
                print('MAP = {}'.format(map_sum/topic_counter))









