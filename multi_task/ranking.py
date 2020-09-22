
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from scipy.spatial import distance

import json
import torch
import os

from multi_task.processing import dataset_metadata
from retrieval.tools import EvalTools


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

    for how in ['euclidean', 'original']:

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


def get_linear_model():
    """ """
    # another way to define a network
    return torch.nn.Sequential(
        torch.nn.Linear(1, 1536),
        torch.nn.ReLU(),
        torch.nn.Linear(1536, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )


def train_model(batch_size=512, lr=0.001, parent_dir_path='/nfs/trec_car/data/entity_ranking/multi_task_data_by_query/'):
    """ """
    train_dir_path = parent_dir_path + 'train_data/'
    dev_dir_path = parent_dir_path + 'dev_data/'

    passage_qrels = dataset_metadata['passage_' + 'dev'][1]
    entity_qrels = dataset_metadata['entity_' + 'dev'][1]

    # ==== Build training data ====
    train_input_list = []
    train_pred_list = []
    for train_query_path in [train_dir_path + f for f in os.listdir(train_dir_path) if 'data.json' in f]:

        query_dict = get_dict_from_json(path=train_query_path)
        query = query_dict['query']['query_id']
        query_cls = query_dict['query']['cls_token']

        for doc_id in query_dict['passage'].keys():
            doc_cls = query_dict['passage'][doc_id]['cls_token']
            relevant = query_dict['passage'][doc_id]['relevant']

            input = query_cls + doc_cls
            train_input_list.append(input)
            train_pred_list.append([relevant])

    train_dataset = TensorDataset(torch.tensor(train_input_list), torch.tensor(train_pred_list))
    train_data_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

    # ==== Build dev data ====
    dev_input_list = []
    dev_pred_list = []
    for dev_query_path in [dev_dir_path + f for f in os.listdir(dev_dir_path) if 'data.json' in f]:

        query_dict = get_dict_from_json(path=dev_query_path)
        query = query_dict['query']['query_id']
        query_cls = query_dict['query']['cls_token']

        for doc_id in query_dict['passage'].keys():
            doc_cls = query_dict['passage'][doc_id]['cls_token']
            relevant = query_dict['passage'][doc_id]['relevant']

            input = query_cls + doc_cls
            dev_input_list.append(input)
            dev_pred_list.append([relevant])

    dev_dataset = TensorDataset(torch.tensor(dev_input_list), torch.tensor(dev_pred_list))
    dev_data_loader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=batch_size)

    # ==== Model setup ====

    model = get_linear_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

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

    # ========================================
    #               Training
    # ========================================

    loss_total = 0
    model.train()
    print(len(train_data_loader))
    for i, train_batch in enumerate(train_data_loader):
        model.zero_grad()


        break




