
from metadata import CarPassagePaths, CarEntityPaths
from retrieval.tools import SearchTools
from torch.utils.data import TensorDataset

import pandas as pd
import torch

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

    for dataset_name, dataset_paths in dataset_metadata.items():
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

        # Read run files
        i = 0
        data = []
        input_ids_list = []
        i_list = []
        with open(run_path, 'r') as f:
            for line in f:
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
                row = [i, dataset_name, run_path, dataset_type, query, doc_id, rank, score, relevant]
                data.append(row)
                i += 1
                if i % print_intervals == 0:
                    print("-- {} --".format(i))
                    print(row)

                # Append tensor data.
                input_ids_list.append(input_ids)
                i_list.append([i])

        # --- Write data to files ---

        # Data.
        parquet_path = dir_path + '_' + dataset_name + '.parquet'
        columns = ['i', 'dataset_name', 'run_path', 'dataset_type', 'query', 'doc_id', 'rank', 'score', 'relevant']
        pd.DataFrame(data, columns=columns).to_parquet(parquet_path)

        # Torch dataset.
        dataset = TensorDataset(torch.tensor(i_list), torch.tensor(input_ids_list))
        tensor_path = dir_path + '_' + dataset_name + '_input_data_.pt'
        # Save tensor dataset to tensor_path.
        print('saving tensor to: {}'.format(tensor_path))
        torch.save(obj=dataset, f=tensor_path)

