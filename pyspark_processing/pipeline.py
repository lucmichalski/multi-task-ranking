
from utils.trec_car_tools import iter_pages
from protocol_buffers import document_pb2
from document_parsing.trec_car_parsing import TrecCarParser

from pyspark.sql.types import BinaryType, StringType, ArrayType
from pyspark.sql.functions import udf, row_number, monotonically_increasing_id, col, collect_list, concat_ws
from pyspark.sql import SparkSession, Window
from pyspark_processing.building_qrels import build_synthetic_qrels

import pandas as pd
import pickle
import time
import json
import os


def write_to_parquet(data, dir_path, dataset, chunk):
    """ write data chunks to parquet """
    parquet_path = dir_path + 'page_dataset_' + str(dataset) + '_chunk_' + str(chunk) + '.parquet'
    columns = ['page_id', 'dataset', 'page_bytearray']
    pd.DataFrame(data, columns=columns).to_parquet(parquet_path)


def write_to_parquet_content(data, dir_path, chunk):
    """ write data chunks to parquet """
    parquet_path = dir_path + 'paragraph_data_chunk_' + str(chunk) + '.parquet'
    columns = ['content_id', 'content_type', 'doc_id', 'dataset', 'content_bytearray']
    pd.DataFrame(data, columns=columns).to_parquet(parquet_path)


def write_pages_data_to_dir(read_path, dir_path, num_pages=1, dataset='unprocessedAllButBenchmark', chunks=100000,
                            print_intervals=100, write_output=False):
    """ Reads TREC CAR cbor file and writes chunks of data to parquet files in 'dir_path'. """
    # create new dir to store data chunks
    if (os.path.isdir(dir_path) == False) and write_output:
        print('making dir: {}'.format(dir_path))
        os.mkdir(dir_path)

    chunk = 0
    pages_data = []
    with open(read_path, 'rb') as f:
        t_start = time.time()
        for i, page in enumerate(iter_pages(f)):

            # stops when 'num_pages' processed
            if i >= num_pages:
                break

            # add bytearray of trec_car_tool.Page object
            pages_data.append([page.page_id, dataset, bytearray(pickle.dumps(page))])

            # write data chunk to file
            if ((i + 1) % chunks == 0) and (i != 0 or num_pages == 1):
                if write_output:
                    print('WRITING TO FILE: {}'.format(i))
                    write_to_parquet(data=pages_data, dir_path=dir_path, dataset=dataset, chunk=chunk)

                    # begin new list
                    pages_data = []
                    chunk += 1

            # prints update at 'print_pages' intervals
            if (i % print_intervals == 0):
                print('----- STEP {} -----'.format(i))
                time_delta = time.time() - t_start
                print('time elapse: {} --> time / page: {}'.format(time_delta, time_delta / (i + 1)))

    if write_output and (len(pages_data) > 0):
        print('WRITING FINAL FILE: {}'.format(i))
        write_to_parquet(data=pages_data, dir_path=dir_path, dataset=dataset, chunk=chunk)

    time_delta = time.time() - t_start
    print('PROCESSED DATA: {} --> processing time / page: {}'.format(time_delta, time_delta / (i + 1)))


def run_pyspark_pipeline(dir_path, spark, cores, out_path):
    """ Reads parquet files from 'dir_path' and parses trec_car_tools.Page object to create protobuf with entity
    linking. """

    print('start preprocessin')
    start_preprocess = time.time()

    # Reads parquet files from 'dir_path' - each row is a TREC CAR pages.
    df_in = spark.read.parquet(dir_path)
    df_in.printSchema()
    num_partitions = df_in.rdd.getNumPartitions()
    print("Number of default partitions: {}".format(num_partitions))

    print('end preprocess')
    end_preprocess = time.time()
    print("*** preprocess time: {:.2f}s ***".format(end_preprocess - start_preprocess))

    print('start pyspark_processing job')
    start_pyspark_job = time.time()

    if num_partitions < cores * 4:
        print('repartitioning df')
        df_in = df_in.repartition(cores*4)
        print("Number of partitions should equal 4*cores --> {}".format(df_in.rdd.getNumPartitions()))

    @udf(returnType=BinaryType())
    def parse_udf(page_bytearray):
        # Parses trec_car_tools.Page object to create protobuf with entity linking.
        page = pickle.loads(page_bytearray)
        tp = TrecCarParser()
        doc = tp.parse_page_to_protobuf(page=page)
        doc_bytearray = pickle.dumps(doc.SerializeToString())
        return doc_bytearray

    # Add index to DF.
    df_parse = df_in.withColumn("doc_bytearray", parse_udf("page_bytearray"))
    df_parse = df_parse.withColumn(
        "index",
        row_number().over(Window.orderBy(monotonically_increasing_id())) - 1
    )
    df_parse.write.parquet(out_path)

    print('end pyspark_processing job')
    end_pyspark_job = time.time()
    print("*** pyspark_processing job time: {:.2f}s ***".format(end_pyspark_job - start_pyspark_job))


def build_qrels(spark, parquet_path, qrels_path, doc_count=1000, qrels_type='tree'):
    """ Build qrels (tree or hierarchical) from 'doc_count' number of preprocessed document data. """
    # Read processed df - each row in a TREC CAR Page and preprocessed protobuf message.
    df = spark.read.parquet(parquet_path)

    # Sample preprocessed data.
    fraction = (doc_count / df.count()) + 0.001
    df_sample = df.sample(withReplacement=False, fraction=fraction)

    # Unpack list of preprocessed data.
    doc_bytearray_list = df_sample.limit(doc_count).select('doc_bytearray').collect()
    document_list = [document_pb2.Document().FromString(pickle.loads(doc_bytearray[0])) for doc_bytearray in doc_bytearray_list]

    # Build qrels (tree or hierarchical).
    build_synthetic_qrels(document_list=document_list, path=qrels_path, qrels_type=qrels_type)


def write_content_data_to_dir(spark, read_path, dir_path, num_contents=1, chunks=10000, write_output=True):
    """ Create document content parquet DF by unpacking preprocessed document data."""
    # Create new dir to store data chunks
    if (os.path.isdir(dir_path) == False) and write_output:
        print('making dir: {}'.format(dir_path))
        os.mkdir(dir_path)

    # Read preprocessed document data.
    df = spark.read.parquet(read_path)
    n = int(df.select("index").rdd.max()[0])
    content_data = []
    chunk = 0
    t_start = time.time()
    # Write chunks of data to files.
    for i in range(0, n+1, chunks):

        # stops when 'num_pages' processed
        if i >= num_contents:
            break

        for df_doc in df.where(df.index.between(i, i + chunks)).collect():
            doc_id = df_doc[0]
            dataset = df_doc[1]
            doc = document_pb2.Document().FromString(pickle.loads(df_doc[3]))
            for doc_content in doc.document_contents:
                # add bytearray of trec_car_tool.Page object
                content_data.append([str(doc_content.content_id),
                                     str(doc_content.content_type),
                                     doc_id,
                                     dataset,
                                     bytearray(pickle.dumps(doc_content.SerializeToString()))])

        if write_output:
            print('----- STEP {} -----'.format(i))
            time_delta = time.time() - t_start
            print('time elapse: {} --> time / page: {}'.format(time_delta, time_delta / (i + 1)))
            write_to_parquet_content(data=content_data, dir_path=dir_path, chunk=chunk)

            # begin new list
            content_data = []
            chunk += 1

    if write_output and (len(content_data) > 0):
        print('WRITING FINAL FILE: {}'.format(i))
        write_to_parquet_content(data=content_data, dir_path=dir_path, chunk=chunk)

    time_delta = time.time() - t_start
    print('PROCESSED DATA: {} --> processing time / page: {}'.format(time_delta, time_delta / (i + 1)))


def add_entity_context_to_pages(spark, pages_path, out_path):
    """ Add entity context to Pages. """
    df = spark.read.parquet(pages_path)

    @udf(returnType=StringType())
    def get_desc(doc_bytearray):
        doc = document_pb2.Document().FromString(pickle.loads(doc_bytearray))
        try:
            return '{}: {}.'.format(doc.doc_name, doc.document_contents[0].text.split(".")[0])
        except:
            return '{}: .'.format(doc.doc_name)

    @udf(returnType=StringType())
    def get_first_para(doc_bytearray):
        doc = document_pb2.Document().FromString(pickle.loads(doc_bytearray))
        try:
            return str(doc.document_contents[0].text)
        except:
            return ""

    @udf(returnType=ArrayType(StringType()))
    def get_top_6_ents(doc_bytearray):
        synthetic_entity_link_totals = document_pb2.Document().FromString(
            pickle.loads(doc_bytearray)).synthetic_entity_link_totals
        link_counts = []
        for synthetic_entity_link_total in synthetic_entity_link_totals:
            entity_id = str(synthetic_entity_link_total.entity_id)
            count = sum([i.frequency for i in synthetic_entity_link_total.anchor_text_frequencies])
            link_counts.append((entity_id, count))
        return [i[0] for i in sorted(link_counts, key=lambda x: x[1], reverse=True)][:5]

    df_desc = df.withColumn("doc_desc", get_desc("doc_bytearray"))
    df_desc_first_para = df_desc.withColumn("first_para", get_first_para("doc_bytearray"))

    df_desc_first_para_ents = df_desc_first_para.withColumn("top_ents", get_top_6_ents("doc_bytearray"))

    doc_desc_df = df_desc_first_para_ents.select(col("page_id").alias("key_id"), "doc_desc")
    doc_top_ents = df_desc_first_para_ents.select("page_id", "first_para", explode("top_ents").alias("key_id"))

    df_join = doc_top_ents.join(doc_desc_df, on=['key_id'], how='left')

    df_group = df_join.groupby("page_id", "first_para").agg(concat_ws(" ", collect_list("doc_desc")).alias("context"))

    df_group.write.parquet(out_path)


def build_entity_context_json(data_path, run_path, out_path=None):
    """ build context json for entities in run. """
    # BUILD CONTEXT
    if out_path == None:
        run_path + '.context.json'

    print('Reading data from: {}'.format(data_path))
    df = spark.read.parquet(data_path)

    print('Building list of page ids in run file.')
    page_ids = []
    with open(run_path, 'r') as f:
        for i, line in enumerate(f):
            page_ids.append(line.split()[2])
    print('Formatting runs for pyspark.')
    page_ids = [[i] for i in list(set(page_ids))]

    print('Building page ids DF.')
    df_run = spark.createDataFrame(page_ids, ["page_id"])
    print('Joining and collecting data.')
    df_join = df_run.join(df, on=["page_id"], how='left').collect()

    print('Building context dict.')
    entities_dict = {}
    for i in df_join:
        entities_dict[i[0]] = {
            'first_para': i[1],
            'top_ents': i[2]
        }

    print('Writing context dict to file: {}'.format(out_path))
    with open(out_path, 'w') as fp:
        json.dump(entities_dict, fp, indent=4)


def add_paragraph_context():
    return

if __name__ == '__main__':

    # exp_time = str(time.time())
    # dir_path = '/nfs/trec_car/data/test_entity/data_in_with_dataset_{}/'.format(exp_time)
    # num_pages = 60000000
    # chunks = 50000
    # print_intervals = 10000
    #
    # read_paths = [
    #     '/nfs/trec_car/entity_processing/entity-linking-with-pyspark_processing/data/benchmark_data/test200.cbor',
    #     '/nfs/trec_car/entity_processing/entity-linking-with-pyspark_processing/data/benchmark_data/benchmarkY2-test.cbor',
    #     '/nfs/trec_car/entity_processing/entity-linking-with-pyspark_processing/data/benchmark_data/benchmarkY1-train.cbor',
    #     '/nfs/trec_car/entity_processing/entity-linking-with-pyspark_processing/data/benchmark_data/benchmarkY1-test.cbor',
    #     '/nfs/trec_car/data/pages/unprocessedAllButBenchmark.Y2.cbor'
    # ]
    # datasets = [
    #     'test200',
    #     'benchmarkY2-test',
    #     'benchmarkY1-train',
    #     'benchmarkY1-test',
    #     'unprocessedAllButBenchmark'
    # ]
    #
    # for read_path, dataset in zip(read_paths, datasets):
    #     print('================================')
    #     print(dataset, read_path)
    #     write_pages_data_to_dir(read_path=read_path,
    #                             dir_path=dir_path,
    #                             num_pages=num_pages,
    #                             dataset=dataset,
    #                             chunks=chunks,
    #                             print_intervals=print_intervals,
    #                             write_output=True)

    spark_drive_gbs = 100
    spark_executor_gbs = 2
    cores = 20

    print('\n//////// RUNNING WITH CORES {} //////////'.format(cores))
    spark = SparkSession.\
        builder\
        .appName('trec_car_spark')\
        .master('local[{}]'.format(cores)) \
        .config("spark.driver.memory", '{}g'.format(spark_drive_gbs)) \
        .config("spark.executor.memory", '{}g'.format(spark_executor_gbs)) \
        .config("spark.driver.maxResultSize", '{}g'.format(spark_drive_gbs)) \
        .getOrCreate()

    print("Settings:")
    print(spark.sparkContext._conf.getAll())

    # dir_path = '/nfs/trec_car/data/test_entity/data_in_with_dataset_1591126867.8381264'
    # out_path = '/nfs/trec_car/data/test_entity/full_data_v3_with_datasets/'
    # run_pyspark_pipeline(dir_path=dir_path,
    #                      spark=spark,
    #                      cores=cores,
    #                      out_path=out_path)

    read_path = '/nfs/trec_car/data/test_entity/full_data_v3_with_datasets/'
    dir_path = '/nfs/trec_car/data/test_entity/full_data_v3_with_datasets_contents_data_dir/'
    write_path = '/nfs/trec_car/data/test_entity/full_data_v3_with_datasets_contents_v4/'
    para_list_path = '/nfs/trec_car/data/test_entity/true_para_id_list.txt'
    write_content_data_to_dir(spark=spark,
                              read_path=read_path,
                              dir_path=dir_path,
                              write_path=write_path,
                              para_list_path=para_list_path,
                              num_contents=10000000000000,
                              chunks=50000,
                              write_output=True)

