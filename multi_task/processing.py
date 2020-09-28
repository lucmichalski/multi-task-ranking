
from metadata import CarPassagePaths, CarEntityPaths
from retrieval.tools import SearchTools
from learning.models import BertCLS, BertMultiTaskRanker

import pandas as pd
import torch
import time
import json
import os
import re

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertModel, BertPreTrainedModel


# Metadata.
dataset_metadata = {
    'entity_train':
        (
        '/nfs/trec_car/data/entity_ranking/multi_task_data/entity_train_all_queries_BM25_1000.run',
        '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_train_data/benchmarkY1_train_entity.qrels'),

    'entity_dev':
        ('/nfs/trec_car/data/entity_ranking/multi_task_data/entity_dev_all_queries_BM25_1000.run',
         '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity.qrels'),

    'entity_test':
        ('/nfs/trec_car/data/entity_ranking/multi_task_data/entity_test_all_queries_BM25_1000.run',
         '/nfs/trec_car/data/entity_ranking/testY1_hierarchical_entity_data/testY1_hierarchical_entity.qrels'),

    'passage_train':
        (
        '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_passage_train_data/benchmarkY1_train_passage_1000.run',
        '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_passage_train_data/benchmarkY1_train_passage.qrels'),

    'passage_dev':
        ('/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_passage_dev_data/benchmarkY1_dev_passage_1000.run',
         '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_passage_dev_data/benchmarkY1_dev_passage.qrels'),

    'passage_test':
        ('/nfs/trec_car/data/entity_ranking/testY1_hierarchical_passage_data/testY1_hierarchical_passage_1000.run',
         '/nfs/trec_car/data/entity_ranking/testY1_hierarchical_passage_data/testY1_hierarchical_passage.qrels')
}

class MultiTaskDataset():

    def __init__(self, dataset_metadata=dataset_metadata):

        self.dataset_metadata = dataset_metadata
        # Dataset counters.
        self.content_i = 0
        self.query_i = -1
        self.chunk_i = 0
        # Chunk data.
        self.content_data = []
        self.bert_i_list = []
        self.bert_input_ids_list = []
        self.query_i_list = []
        self.query_input_ids_list = []


    def __reset_chunk(self):
        """ """
        self.content_data = []
        self.bert_input_ids_list = []
        self.bert_i_list = []
        self.query_i_list = []
        self.query_input_ids_list = []


    def __write_to_pt_file(self, tensor_path, list_1, list_2):
        """ """
        dataset = TensorDataset(torch.tensor(list_1), torch.tensor(list_2))
        print('saving tensor to: {}'.format(tensor_path))
        torch.save(obj=dataset, f=tensor_path)


    def __make_dir(self, dir_path):
        """ """
        if (os.path.isdir(dir_path) == False):
            print('making dir: {}'.format(dir_path))
            os.mkdir(dir_path)


    def __get_content_dir_path(self, dir_path, dataset_name):
        """ """
        return dir_path + dataset_name + '_content_data/'


    def __get_bert_dir_path(self, dir_path, dataset_name):
        """ """
        return dir_path + dataset_name + '_bert_data/'


    def __get_bert_processed_dir_path(self, dir_path, dataset_name):
        """ """
        return dir_path + dataset_name + '_bert_data_processed/'


    def __get_query_dir_path(self, dir_path, dataset_name):
        """ """
        return dir_path + dataset_name + '_bert_query_data/'


    def __get_query_processed_dir_path(self, dir_path, dataset_name):
        """ """
        return dir_path + dataset_name + '_bert_query_data_processed/'


    def write_chunk(self, dir_path, dataset_name):
        """ """
        print('=== WRITING CHUNK {} ==='.format(self.chunk_i))
        # Content data.
        content_dir_path = self.__get_content_dir_path(dir_path=dir_path, dataset_name=dataset_name)
        self.__make_dir(content_dir_path)
        parquet_path = content_dir_path + 'chunk_{}.parquet'.format(self.chunk_i)
        columns = ['content_i', 'query_i', 'dataset_name', 'run_path', 'dataset_type', 'query', 'doc_id', 'rank', 'score', 'relevant']
        print('saving data to: {}'.format(parquet_path))
        pd.DataFrame(self.content_data, columns=columns).to_parquet(parquet_path)

        # BERT data.
        bert_dir_path = self.__get_bert_dir_path(dir_path=dir_path, dataset_name=dataset_name)
        self.__make_dir(bert_dir_path)
        tensor_path = bert_dir_path + 'chunk_{}.pt'.format(self.chunk_i)
        self.__write_to_pt_file(tensor_path=tensor_path,
                                list_1=self.bert_i_list,
                                list_2=self.bert_input_ids_list)

        # BERT query data.
        bert_query_dir_path = self.__get_query_dir_path(dir_path=dir_path, dataset_name=dataset_name)
        self.__make_dir(bert_query_dir_path)
        tensor_path = bert_query_dir_path + 'chunk_{}.pt'.format(self.chunk_i)
        self.__write_to_pt_file(tensor_path=tensor_path,
                                list_1=self.query_i_list,
                                list_2=self.query_input_ids_list)


    def build_datasets(self, dir_path, max_rank=100, chuck_size=250000, print_intervals=50000):
        """ """

        for dataset_name, dataset_paths in self.dataset_metadata.items():

            # Initialise dataset.
            self.content_i = 0
            self.query_i = -1
            self.chunk_i = 0

            # Reset chunk.
            self.__reset_chunk()

            print('======= {} ======='.format(dataset_name))
            run_path = dataset_paths[0]
            qrels_path = dataset_paths[1]

            # Define ranking type.
            if 'passage' in dataset_name:
                index_path = CarPassagePaths.index
            elif 'entity' in dataset_name:
                index_path = CarEntityPaths.index
            else:
                index_path = None
                print("NOT VALID DATASET")

            # Define dataset type.
            if 'train' in dataset_name:
                dataset_type = 'train'
            elif 'dev' in dataset_name:
                dataset_type = 'dev'
            elif 'test' in dataset_name:
                dataset_type = 'test'
            else:
                dataset_type = None
                print("NOT VALID DATASET")

            search_tools = SearchTools(index_path=index_path)
            qrels = search_tools.retrieval_utils.get_qrels_binary_dict(qrels_path=qrels_path)
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            start_time = time.time()

            # Read run files
            past_query = ''
            with open(run_path, 'r') as f:
                for line in f:
                    # Unpack run line.
                    query_encoded, _, doc_id, rank, score, _ = search_tools.retrieval_utils.unpack_run_line(line)

                    if int(rank) <= max_rank:
                        # Decode query.
                        try:
                            query = search_tools.decode_query_car(q=query_encoded)
                        except ValueError:
                            print(
                                "URL utf-8 decoding did not work with Pyserini's SimpleSearcher.search()/JString: {}".format(query))
                            query = search_tools.process_query_car(q=query_encoded)

                        # Get relevant score
                        try:
                            if doc_id in qrels[query_encoded]:
                                relevant = 1
                            else:
                                relevant = 0
                        except:
                            relevant = 0

                        # Get text and input ids.
                        text_full = search_tools.get_contents_from_docid(doc_id=doc_id)
                        if 'entity' in dataset_name:
                            text = text_full.split('\n')[0]
                        else:
                            text = text_full
                        input_ids = tokenizer.encode(text=text,
                                                     max_length=512,
                                                     add_special_tokens=True,
                                                     pad_to_max_length=True)
                        self.bert_input_ids_list.append(input_ids)
                        self.bert_i_list.append([self.content_i])

                        # Set query id.
                        if past_query != query_encoded:
                            self.query_i += 1
                            query_input_ids = tokenizer.encode(text=query,
                                                               max_length=512,
                                                               add_special_tokens=True,
                                                               pad_to_max_length=True)
                            self.query_i_list.append([self.query_i])
                            self.query_input_ids_list.append(query_input_ids)

                        # Append data.
                        row = [self.content_i, self.query_i, dataset_name, run_path, dataset_type, query_encoded, doc_id, rank, score, relevant]
                        self.content_data.append(row)

                        # Print update.
                        if self.content_i % print_intervals == 0:
                            end_time = time.time()
                            print("-- {} -- dataset time: {:.2f} ---".format(self.content_i, end_time-start_time))
                            print(row)

                        # Write chunk.
                        if (self.content_i % chuck_size == 0) and (self.content_i != 0):
                            self.write_chunk(dir_path=dir_path, dataset_name=dataset_name)
                            self.__reset_chunk()
                            self.chunk_i += 1

                        # Re-set query counter.
                        self.content_i += 1
                        past_query = query_encoded

            #  Write final chunk.
            if len(self.bert_input_ids_list) > 0:
                self.write_chunk(dir_path=dir_path, dataset_name=dataset_name)
                self.__reset_chunk()
                self.chunk_i += 1


    def cls_processing(self, dir_path, batch_size=64):
        """ """
        # Load BERT model to get CLS token.
        model = BertCLS.from_pretrained('bert-base-uncased')
        model.eval()

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

        for dataset_name in self.dataset_metadata.keys():

            bert_dir_path = self.__get_bert_dir_path(dir_path=dir_path, dataset_name=dataset_name)
            query_dir_path = self.__get_query_dir_path(dir_path=dir_path, dataset_name=dataset_name)

            bert_paths = [bert_dir_path + f for f in os.listdir(bert_dir_path)]
            query_paths = [query_dir_path + f for f in os.listdir(query_dir_path)]

            for paths in [bert_paths, query_paths]:
                for read_path in paths:
                    print('processing: {}'.format(read_path))

                    read_dataset = torch.load(read_path)
                    data_loader = DataLoader(read_dataset, sampler=SequentialSampler(read_dataset), batch_size=batch_size)

                    id_list = []
                    cls_tokens = []

                    for i, batch in enumerate(data_loader):
                        b_id_list = batch[0]
                        b_input_ids = batch[1].to(device)
                        with torch.no_grad():
                            b_cls_tokens = model.get_BERT_cls_vector(input_ids=b_input_ids)

                        id_list.append(b_id_list)
                        cls_tokens.append(b_cls_tokens.cpu())

                        if (i + 1) % 10 == 0:
                            print("processed: {} / {}".format(i + 1, len(data_loader)))

                    id_list_tensor = torch.cat(id_list)
                    cls_tokens_tensor = torch.cat(cls_tokens)

                    write_dataset = TensorDataset(id_list_tensor, cls_tokens_tensor)
                    write_dir = dir_path + read_path.split('/')[-2:][0] + '_processed/'
                    if (os.path.isdir(write_dir) == False):
                        print('making dir: {}'.format(dir_path))
                        os.mkdir(write_dir)
                    write_path = write_dir + read_path.split('/')[-1:][0]
                    # Save tensor dataset to tensor_path.
                    print('saving tensor to: {}'.format(write_path))
                    torch.save(obj=write_dataset, f=write_path)

    def __get_query_data(self, dir_path, entity_dataset_name):
        """ """
        pass



    def __get_id_to_cls_map(self, dir_path):
        """ """
        path_list = os.listdir(dir_path)
        path_list.sort(key=lambda f: int(re.sub('\D', '', f)))
        print('ordered files: {}'.format(path_list))

        id_to_cls_map = {}
        for file_name in path_list:
            if file_name[len(file_name) - 3:] == '.pt':
                path = os.path.join(dir_path, file_name)
                print('loadings data from: {}'.format(path))
                dataset = torch.load(path)
                for i, cls_token in dataset:
                    id_to_cls_map[int(i.numpy())] = cls_token

        return id_to_cls_map


    def __get_run_data(self, content_path):
        """ """
        print('reading parquet file: {}'.format(content_path))
        #df = pd.read_parquet(content_path)
        #print(df.head())
        run_data = {}
        for i, row in pd.read_parquet(content_path).itterrows():
            query = row['query']
            query_i = row['query_i']
            content_i = row['content_i']
            if query not in run_data:
                run_data[query] = {
                    'query_i': query_i,
                    'content_i': [content_i]
                }
            else:
                run_data[query]['content_i'].append(content_i)

        return run_data


    def get_query_specific_data(self, dir_path, dataset):
        """ """
        entity_dataset_name = 'entity_' + dataset
        passage_dataset_name = 'passage_' + dataset

        # Content entity data
        entity_content_path = self.__get_content_dir_path(dir_path=dir_path, dataset_name=entity_dataset_name)
        #df_entity_content = pd.read_parquet(entity_content_path)
        entity_run_data = self.__get_run_data(content_path=entity_content_path)
        for key, value in entity_run_data.items():
            print(key, value)
            break
        # print('----- df_entity_content  -------')
        # print(df_entity_content.head())
        # print('--------------------------------')
        # print()
        #
        # # Content passgae data
        # passage_content_path = self.__get_content_dir_path(dir_path=dir_path, dataset_name=passage_dataset_name)
        # print('reading parquet file: {}'.format(passage_content_path))
        # df_passage_content = pd.read_parquet(entity_content_path)
        # print('----- df_entity_content  -------')
        # print(df_entity_content.head())
        # print('--------------------------------')
        # print()
        #
        # #
        # query_i_list = sorted(df_passage_content['query_i'].to_list())
        #
        # # Query dats.
        # query_cls_path = self.__get_query_processed_dir_path(dir_path=dir_path, dataset_name=entity_dataset_name)
        # query_i_to_cls = self.__get_id_to_cls_map(dir_path=query_cls_path)
        #
        # # Entity CLS data
        # entity_cls_path = self.__get_bert_processed_dir_path(dir_path=dir_path, dataset_name=entity_dataset_name)
        # entity_i_to_cls = self.__get_id_to_cls_map(dir_path=entity_cls_path)
        #
        # # Passage CLS data.
        # passage_cls_path = self.__get_bert_processed_dir_path(dir_path=dir_path, dataset_name=passage_dataset_name)
        # passage_i_to_cls = self.__get_id_to_cls_map(dir_path=passage_cls_path)
        #
        # for query_i in query_i_list:
        #     print('=== query_i: {} ==='.format(query_i))
        #     print(query_i_to_cls[query_i])
        #
        #     entity_i_list = sorted(df_entity_content[df_entity_content['query_i'] == query_i]['content_i'].to_list())
        #
        #     passage_i_list = sorted(df_passage_content[df_passage_content['query_i'] == query_i]['content_i'].to_list())
        #
        #     for content_i in entity_i_list:
        #         print('=== content_i: {} ==='.format(content_i))
        #         print(type(content_i))
        #         df_entity_content_query = df_entity_content[df_entity_content['content_i'] == content_i]
        #         print('----- df_entity_content_query  -------')
        #         print(df_entity_content_query.head())
        #         print()
        #         print('----- Entity CLS  -------')
        #         print(entity_i_to_cls[content_i])
        #         print('--------------------------------')
        #
        #         break
        #
        #     for content_i in passage_i_list:
        #         print('=== content_i: {} ==='.format(content_i))
        #         print(type(content_i))
        #         df_passage_content_query = df_passage_content[df_passage_content['content_i'] == content_i]
        #         print('----- df_entity_content_query  -------')
        #         print(df_passage_content_query.head())
        #         print()
        #         print('----- Entity CLS  -------')
        #         print(passage_i_to_cls[content_i])
        #         print('--------------------------------')
        #
        #         break
        #

class MultiTaskDatasetByQuery():

    def __init__(self, dataset_metadata=dataset_metadata):

        self.dataset_metadata = dataset_metadata
        self.cls_id = 0
        self.token_list = []
        self.cls_id_list = []


    def __make_dir(self, dir_path):
        """ """
        if (os.path.isdir(dir_path) == False):
            print('making dir: {}'.format(dir_path))
            os.mkdir(dir_path)


    def get_task_run_and_qrels(self, dataset, task='entity', max_rank=100):
        """ """
        dataset_name = task + '_' + dataset
        dataset_paths = self.dataset_metadata[dataset_name]

        search_tools = SearchTools()
        qrels_path = dataset_paths[1]
        qrels_dict = search_tools.retrieval_utils.get_qrels_binary_dict(qrels_path=qrels_path)

        run_path = dataset_paths[0]
        run_dict = {}
        with open(run_path, 'r') as f:
            for line in f:
                # Unpack run line.
                query, _, doc_id, rank, _, _ = search_tools.retrieval_utils.unpack_run_line(line)

                if int(rank) <= max_rank:
                    if query not in run_dict:
                        run_dict[query] = []
                    run_dict[query].append([doc_id, rank])

        return run_dict, qrels_dict


    def build_dataset_by_query(self, dir_path, max_rank=100, batch_size=64, bi_encode=False, passage_model_path=None,
                               entity_model_path=None):
        """ """

        passage_model = BertMultiTaskRanker.from_pretrained(passage_model_path)
        entity_model = BertMultiTaskRanker.from_pretrained(entity_model_path)

        # Use GPUs if available.
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name())
            # Tell PyTorch to use the GPU.
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU: {}'.format(torch.cuda.get_device_name(0)))
            passage_model.cuda()
            entity_model.cuda()
            device = torch.device("cuda")
        # Otherwise use CPU.
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")

        for dataset in ['dev', 'test', 'train']:

            dataset_dir_path = dir_path + '{}_data/'.format(dataset)
            self.__make_dir(dataset_dir_path)

            entity_run_dict, entity_qrels_dict = self. get_task_run_and_qrels(dataset=dataset, task='entity', max_rank=max_rank)
            passage_run_dict, passage_qrels_dict = self. get_task_run_and_qrels(dataset=dataset, task='passage', max_rank=max_rank)

            entity_query_list = sorted(list(entity_run_dict.keys()))
            passage_query_list = sorted(list(passage_run_dict.keys()))

            assert entity_query_list == passage_query_list

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            search_tools_passage = SearchTools(index_path=CarPassagePaths.index)
            search_tools_entity = SearchTools(index_path=CarEntityPaths.index)

            for query_i, query in enumerate(entity_query_list):
                print("Processing {} ({} / {})".format(query, query_i+1, len(entity_query_list)))

                # Reset query dataset.
                query_dataset = {}
                self.cls_id = 0
                self.cls_id_list = []
                self.token_list = []

                # ======== PROCESS QUERY ========
                query_dataset['query'] = {}
                query_dataset['query']['query_id'] = query
                query_dataset['query']['cls_id'] = self.cls_id

                # Update BERT cls input
                try:
                    query_decoded = search_tools_passage.decode_query_car(q=query)
                except ValueError:
                    print(
                        "URL utf-8 decoding did not work with Pyserini's SimpleSearcher.search()/JString: {}".format(query))
                    query_decoded = search_tools_passage.process_query_car(q=query)

                input_ids = tokenizer.encode(text=query_decoded,
                                             max_length=512,
                                             add_special_tokens=True,
                                             pad_to_max_length=True)

                # Add CLS data.
                self.cls_id_list.append([self.cls_id])
                self.token_list.append(input_ids)
                self.cls_id += 1

                # ======== PROCESS PASSAGE ========
                passage_run_data = passage_run_dict[query]
                query_dataset['passage'] = {}
                for run_data in passage_run_data:
                    doc_id = run_data[0]
                    rank = run_data[1]

                    query_dataset['passage'][doc_id] = {}
                    query_dataset['passage'][doc_id]['rank'] = rank
                    query_dataset['passage'][doc_id]['cls_id'] = self.cls_id
                    if query in passage_qrels_dict:
                        if doc_id in passage_qrels_dict[query]:
                            query_dataset['passage'][doc_id]['relevant'] = 1
                        else:
                            query_dataset['passage'][doc_id]['relevant'] = 0
                    else:
                        query_dataset['passage'][doc_id]['relevant'] = 0

                    text = search_tools_passage.get_contents_from_docid(doc_id=doc_id)
                    if bi_encode == False:
                        input_ids = tokenizer.encode(text=text,
                                                     max_length=512,
                                                     add_special_tokens=True,
                                                     pad_to_max_length=True)
                    else:
                        input_ids = tokenizer.encode(text=query,
                                                     text_pair=text,
                                                     max_length=512,
                                                     add_special_tokens=True,
                                                     pad_to_max_length=True)
                    # Add CLS data.
                    self.cls_id_list.append([self.cls_id])
                    self.token_list.append(input_ids)
                    self.cls_id += 1

                passage_dataset = TensorDataset(torch.tensor(self.cls_id_list), torch.tensor(self.token_list))
                passage_data_loader = DataLoader(passage_dataset, sampler=SequentialSampler(passage_dataset), batch_size=batch_size)

                self.cls_id_list = []
                self.token_list = []

                # ======== PROCESS ENTITY ========
                entity_run_data = entity_run_dict[query]
                query_dataset['entity'] = {}
                for run_data in entity_run_data:
                    doc_id = run_data[0]
                    rank = run_data[1]

                    query_dataset['entity'][doc_id] = {}
                    query_dataset['entity'][doc_id]['rank'] = rank
                    query_dataset['entity'][doc_id]['cls_id'] = self.cls_id
                    if query in entity_qrels_dict:
                        if doc_id in entity_qrels_dict[query]:
                            query_dataset['entity'][doc_id]['relevant'] = 1
                        else:
                            query_dataset['entity'][doc_id]['relevant'] = 0
                    else:
                        query_dataset['entity'][doc_id]['relevant'] = 0

                    text_full = search_tools_entity.get_contents_from_docid(doc_id=doc_id)
                    text = text_full.split('\n')[0]
                    if bi_encode == False:
                        input_ids = tokenizer.encode(text=text,
                                                     max_length=512,
                                                     add_special_tokens=True,
                                                     pad_to_max_length=True)
                    else:
                        input_ids = tokenizer.encode(text=query,
                                                     text_pair=text,
                                                     max_length=512,
                                                     add_special_tokens=True,
                                                     pad_to_max_length=True)
                    # Add CLS data.
                    self.cls_id_list.append([self.cls_id])
                    self.token_list.append(input_ids)
                    self.cls_id += 1

                entity_dataset = TensorDataset(torch.tensor(self.cls_id_list), torch.tensor(self.token_list))
                entity_data_loader = DataLoader(entity_dataset, sampler=SequentialSampler(entity_dataset), batch_size=batch_size)

                id_list = []
                cls_tokens = []
                for batch in passage_data_loader:
                    b_id_list = batch[0]
                    b_input_ids = batch[1].to(device)
                    with torch.no_grad():
                        b_cls_tokens =passage_model.bert.forward(input_ids=b_input_ids)

                    id_list.append(b_id_list)
                    cls_tokens.append(b_cls_tokens[0].cpu())

                for batch in entity_data_loader:
                    b_id_list = batch[0]
                    b_input_ids = batch[1].to(device)
                    with torch.no_grad():
                        b_cls_tokens = entity_model.bert.forward(input_ids=b_input_ids)

                    id_list.append(b_id_list)
                    cls_tokens.append(b_cls_tokens[0].cpu())

                id_list_tensor = torch.cat(id_list).numpy().tolist()
                cls_tokens_tensor = torch.cat(cls_tokens).numpy().tolist()
                cls_map = {}
                for cls_i, cls_token in zip(id_list_tensor, cls_tokens_tensor):
                    cls_map[int(cls_i[0])] = cls_token

                # ======== PROCESS CLS TOKENS ========
                # Add query CLS token.
                query_cls_id = query_dataset['query']['cls_id']
                query_dataset['query']['cls_token'] = cls_map[query_cls_id]

                for doc_id in query_dataset['entity'].keys():
                    entity_cls_id = query_dataset['entity'][doc_id]['cls_id']
                    query_dataset['entity'][doc_id]['cls_token'] = cls_map[entity_cls_id]

                for doc_id in query_dataset['passage'].keys():
                    passage_cls_id = query_dataset['passage'][doc_id]['cls_id']
                    query_dataset['passage'][doc_id]['cls_token'] = cls_map[passage_cls_id]

                if bi_encode == False:
                    query_json_path = dataset_dir_path + '{}_data_ranker.json'.format(query_i)
                else:
                    query_json_path = dataset_dir_path + '{}_data_bi_encode_ranker.json'.format(query_i)
                with open(query_json_path, 'w') as f:
                    json.dump(query_dataset, f, indent=4)



def create_extra_queries(dir_path, dataset_metadata=dataset_metadata):
    """ """
    for dataset in ['dev', 'train', 'test']:

        entity_qrels_path = dataset_metadata['entity_' + dataset][1]
        passage_qrels_path = dataset_metadata['passage_' + dataset][1]

        search_tools = SearchTools(index_path=None)

        entity_queries = set(search_tools.retrieval_utils.get_qrels_binary_dict(qrels_path=entity_qrels_path).keys())
        passage_queries = set(search_tools.retrieval_utils.get_qrels_binary_dict(qrels_path=passage_qrels_path).keys())

        def write_topics(dir_path, original_queries, missing_queries, dataset, ranking_type):
            """ """
            if len(missing_queries) > 0:
                print('{} missing queries for {} {} dataset'.format(len(missing_queries), ranking_type, dataset))
                missing_queries_path = dir_path + ranking_type + '_' + dataset + '_all_queries.topics'
                all_queries = sorted(list(original_queries) + list(missing_queries))
                print('-> all_queries: {}'.format(len(all_queries)))
                with open(missing_queries_path, 'w') as f:
                    for q in list(all_queries):
                        f.write(q + '\n')
            else:
                print('No missing queries for {} {} dataset (total: {})'.format(ranking_type, dataset, len(original_queries)))

        # Build all entity queries.
        missing_entity_queries = passage_queries - entity_queries
        write_topics(dir_path=dir_path,
                     original_queries=entity_queries,
                     missing_queries=missing_entity_queries,
                     dataset=dataset,
                     ranking_type='entity')

        # Build all passage queries.
        missing_passage_queries = entity_queries - passage_queries
        write_topics(dir_path=dir_path,
                     original_queries=passage_queries,
                     missing_queries=missing_passage_queries,
                     dataset=dataset,
                     ranking_type='passage')


