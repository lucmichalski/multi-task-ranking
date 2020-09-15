
from metadata import CarPassagePaths, CarEntityPaths
from retrieval.tools import SearchTools
from learning.models import BertCLS

import pandas as pd
import torch
import time
import os

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertModel, BertPreTrainedModel


# Metadata.
dataset_metadata = {
    # 'entity_train':
    #     (
    #     '/nfs/trec_car/data/entity_ranking/multi_task_data/entity_train_all_queries_BM25_1000.run',
    #     '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_train_data/benchmarkY1_train_entity.qrels'),

    'entity_dev':
        ('/nfs/trec_car/data/entity_ranking/multi_task_data/entity_dev_all_queries_BM25_1000.run',
         '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity.qrels'),

    # 'entity_test':
    #     ('/nfs/trec_car/data/entity_ranking/multi_task_data/entity_test_all_queries_BM25_1000.run',
    #      '/nfs/trec_car/data/entity_ranking/testY1_hierarchical_entity_data/testY1_hierarchical_entity.qrels'),
    #
    # 'passage_train':
    #     (
    #     '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_passage_train_data/benchmarkY1_train_passage_1000.run',
    #     '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_passage_train_data/benchmarkY1_train_passage.qrels'),
    #
    # 'passage_dev':
    #     ('/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_passage_dev_data/benchmarkY1_dev_passage_1000.run',
    #      '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_passage_dev_data/benchmarkY1_dev_passage.qrels'),
    #
    # 'passage_test':
    #     ('/nfs/trec_car/data/entity_ranking/testY1_hierarchical_passage_data/testY1_hierarchical_passage_1000.run',
    #      '/nfs/trec_car/data/entity_ranking/testY1_hierarchical_passage_data/testY1_hierarchical_passage.qrels')
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


    def __get_query_dir_path(self, dir_path, dataset_name):
        """ """
        return dir_path + dataset_name + '_bert_query_data/'


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


    def build_datasets(self, dir_path, chuck_size=250000, print_intervals=50000):
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
            query_paths = [bert_dir_path + f for f in os.listdir(query_dir_path)]

            for paths in [bert_paths, query_paths]:
                for path in paths:

                    in_dataset = torch.load(path)
                    data_loader = DataLoader(in_dataset, sampler=SequentialSampler(in_dataset), batch_size=batch_size)

                    id_list = []
                    cls_tokens = []

                    for batch in data_loader:
                        b_id_list = batch[0]
                        b_input_ids = batch[1].to(device)
                        with torch.no_grad():
                            b_cls_tokens = model.get_BERT_cls_vector(input_ids=b_input_ids)

                        id_list.append(b_id_list)
                        cls_tokens.append(b_cls_tokens.cpu())

                    id_list_tensor = torch.cat(cls_tokens)
                    cls_tokens_tensor = torch.cat(cls_tokens)

                    out_dataset = TensorDataset(id_list_tensor, cls_tokens_tensor)
                    tensor_dir = dir_path + path.split('/')[-2:][0] + '_processed/'
                    if (os.path.isdir(tensor_dir) == False):
                        print('making dir: {}'.format(dir_path))
                        os.mkdir(tensor_dir)
                    tensor_path = tensor_dir + path.split('/')[-1:]
                    # Save tensor dataset to tensor_path.
                    print('saving tensor to: {}'.format(tensor_path))
                    torch.save(obj=out_dataset, f=tensor_path)


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