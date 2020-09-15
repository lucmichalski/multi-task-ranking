
from metadata import CarPassagePaths, CarEntityPaths
from retrieval.tools import SearchTools
from learning.models import BertCLS

import pandas as pd
import torch
import time

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertModel, BertPreTrainedModel


# Metadata.
dataset_metadata = {
    'entity_train':
        (
        '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_train_data/benchmarkY1_train_entity_1000.run',
        '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_train_data/benchmarkY1_train_entity.qrels'),

    'entity_dev':
        ('/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity_1000.run',
         '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity.qrels'),

    'entity_test':
        ('/nfs/trec_car/data/entity_ranking/testY1_hierarchical_entity_data/testY1_hierarchical_entity_1000.run',
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


def build_datasets(dir_path, print_intervals=100000, dataset_metadata=dataset_metadata):
    """ """

    for dataset_name, dataset_paths in dataset_metadata.items():

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
        row_i = 0
        query_i = -1
        data = []

        query_i_list = []
        query_input_ids_list = []

        input_ids_list = []
        row_i_list = []

        past_query = ''
        with open(run_path, 'r') as f:
            for line in f:
                query_encoded, _, doc_id, rank, score, _ = search_tools.retrieval_utils.unpack_run_line(line)

                # Set query id.
                if past_query != query_encoded:
                    query_i += 1
                    query_input_ids = tokenizer.encode(text=text,
                                                       max_length=512,
                                                       add_special_tokens=True,
                                                       pad_to_max_length=True)
                    query_i_list.append([query_i])
                    query_input_ids_list.append(query_input_ids)

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

                # Get text.
                text_full = search_tools.get_contents_from_docid(doc_id=doc_id)
                if 'entity' in dataset_name:
                    text = text_full.split('\n')[0]
                else:
                    text = text_full

                input_ids = tokenizer.encode(text=text,
                                             max_length=512,
                                             add_special_tokens=True,
                                             pad_to_max_length=True)

                # Append data.
                row = [row_i, query_i, dataset_name, run_path, dataset_type, query_encoded, doc_id, rank, score, relevant]
                data.append(row)

                # Append tensor data.
                input_ids_list.append(input_ids)
                row_i_list.append([row_i])

                # Print update.
                if row_i % print_intervals == 0:
                    end_time = time.time()

                    print("-- {} -- dataset time: {:.2f} ---".format(row_i, end_time-start_time))
                    print(row)
                row_i += 1

                # Re-set query counter.
                past_query = query_encoded

        # --- Write data to files ---

        # Data.
        parquet_path = dir_path + dataset_name + '_data.parquet'
        columns = ['row_i', 'query_i', 'dataset_name', 'run_path', 'dataset_type', 'query', 'doc_id', 'rank', 'score', 'relevant']
        print('saving data to: {}'.format(parquet_path))
        pd.DataFrame(data, columns=columns).to_parquet(parquet_path)

        # Torch E/P dataset.
        def write_to_pt_file(tensor_path, list_1, list_2):
            dataset = TensorDataset(torch.tensor(list_1), torch.tensor(list_2))
            print('saving tensor to: {}'.format(tensor_path))
            torch.save(obj=dataset, f=tensor_path)

        tensor_path = dir_path + dataset_name + '_bert_data.pt'
        write_to_pt_file(tensor_path=tensor_path,
                         list_1=row_i_list,
                         list_2=input_ids_list)

        tensor_path = dir_path + dataset_name + '_bert_query_data.pt'
        write_to_pt_file(tensor_path=tensor_path,
                         list_1=query_i_list,
                         list_2=query_input_ids_list)


def cls_processing(dir_path, batch_size=64, dataset_metadata=dataset_metadata):
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


    for dataset_name in dataset_metadata.keys():

        tensor_path = dir_path + dataset_name + '_bert_data.pt'
        dataset = torch.load(tensor_path)
        data_loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

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

        dataset = TensorDataset(id_list_tensor, cls_tokens_tensor)
        tensor_path = dir_path  + dataset_name + '_bert_cls_tokens.pt'
        # Save tensor dataset to tensor_path.
        print('saving tensor to: {}'.format(tensor_path))
        torch.save(obj=dataset, f=tensor_path)