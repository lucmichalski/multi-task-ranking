
from pyspark.sql.types import BinaryType, StringType, ArrayType
from pyspark.sql.functions import udf, row_number, monotonically_increasing_id, col, collect_list, concat_ws, explode
from pyspark.sql import SparkSession, Window

from protocol_buffers import document_pb2

import pickle
import json

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


def build_passage_to_entity_maps(content_path, spark, max_rank, dir_path, dataset_metadata=dataset_metadata):
    """" """
    df = spark.read.parquet(content_path)
    df.printSchema()

    @udf(returnType=ArrayType(StringType()))
    def get_ents(content_bytearray):
        synthetic_entity_links = document_pb2.DocumentContent().FromString(
            pickle.loads(content_bytearray)).synthetic_entity_links
        entity_links = []
        for synthetic_entity_link in synthetic_entity_links:
            entity_links.append(str(synthetic_entity_link.entity_id))
        return entity_links

    df_entity = df.withColumn("entities", get_ents("content_bytearray"))
    df_entity.printSchema()

    for dataset in ['dev']:
        dateset_dir = dir_path + '{}_data/'.format(dataset)
        passage_name = 'passage' + '_{}'.format(dataset)
        passage_path = dataset_metadata[passage_name][0]
        print('================================')
        print('Building passage->entity mappings for {}: {}'.format(dataset, passage_path))
        run_dict = {}
        doc_ids_list = []
        with open(passage_path, 'r') as f:
            for line in f:

                query = line.split()[0]
                doc_id = line.split()[2]
                rank = int(line.split()[3])

                if rank <= max_rank:

                    if query not in run_dict:
                        run_dict[query] = []
                    run_dict[query].append(doc_id)
                    doc_ids_list.append(doc_id)

        query_list = sorted(list(run_dict.keys()))

        doc_ids_list = list(set(doc_ids_list))
        print("doc_ids_list len = {}".format(len(doc_ids_list)))
        dataset_df = df_entity[df_entity['content_id'].isin(doc_ids_list)].select("content_id", "entities")
        print("dataset_map len = {}".format(dataset_df.count()))
        print(dataset_df.head())

        dataset_dict = {}
        for row in dataset_df.collect():
            dataset_dict[row[0]] = row[1]

        print("dataset_dict len = {}".format(len(dataset_dict)))

        write_json_path = dateset_dir + 'passage_to_entity.json'
        print('writing to: {}'.format(write_json_path))
        with open(write_json_path, 'w') as f:
            json.dump(dataset_dict, f, indent=4)
