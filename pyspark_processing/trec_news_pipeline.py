
from utils.trec_car_tools import iter_pages
from protocol_buffers import document_pb2
from document_parsing.trec_news_parsing import TrecNewsParser

from pyspark.sql.types import BinaryType, StringType, ArrayType
from pyspark.sql.functions import udf, row_number, monotonically_increasing_id, col, collect_list, concat_ws, explode
from pyspark.sql import SparkSession, Window
from pyspark_processing.building_qrels import build_synthetic_qrels
from collections import Counter

import pandas as pd
import pickle
import time
import json
import os


def write_to_parquet(data, dir_path,  chunk):
    """ write data chunks to parquet """
    parquet_path = dir_path + 'article_data_chunk_' + str(chunk) + '.parquet'
    columns = ['article_id', 'article_bytearray']
    pd.DataFrame(data, columns=columns).to_parquet(parquet_path)


def write_article_data_to_dir(read_path, dir_path, num_pages=1, chunks=100000, print_intervals=100, write_output=False):
    """ Reads TREC news json file and writes chunks of data to parquet files in 'dir_path'. """
    # create new dir to store data chunks
    if (os.path.isdir(dir_path) == False) and write_output:
        print('making dir: {}'.format(dir_path))
        os.mkdir(dir_path)

    chunk = 0
    article_data = []
    with open(read_path, encoding="utf-8") as f:
        t_start = time.time()
        for i, line in enumerate(f):

            article_json = json.loads(line)
            article = TrecNewsParser().build_article_from_json(article_json)

            # stops when 'num_pages' processed
            if i >= num_pages:
                break

            # add bytearray of trec_car_tool.Page object
            article_data.append([article['id'], bytearray(pickle.dumps(article))])

            # write data chunk to file
            if ((i + 1) % chunks == 0) and (i != 0 or num_pages == 1):
                if write_output:
                    print('WRITING TO FILE: {}'.format(i))
                    write_to_parquet(data=article_data, dir_path=dir_path, chunk=chunk)

                    # begin new list
                    article_data = []
                    chunk += 1

            # prints update at 'print_pages' intervals
            if (i % print_intervals == 0):
                print('----- STEP {} -----'.format(i))
                time_delta = time.time() - t_start
                print('time elapse: {} --> time / page: {}'.format(time_delta, time_delta / (i + 1)))

    if write_output and (len(article_data) > 0):
        print('WRITING FINAL FILE: {}'.format(i))
        write_to_parquet(data=article_data, dir_path=dir_path, chunk=chunk)

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
    def parse_udf(article_bytearray):
        # Parses trec_car_tools.Page object to create protobuf with entity linking.
        article = pickle.loads(article_bytearray)
        tp = TrecNewsParser()
        doc = tp.parse_article_to_protobuf(article=article)
        doc_bytearray = pickle.dumps(doc.SerializeToString())
        return doc_bytearray

    # Add index to DF.
    df_parse = df_in.withColumn("doc_bytearray", parse_udf("article_bytearray"))
    df_parse = df_parse.withColumn(
        "index",
        row_number().over(Window.orderBy(monotonically_increasing_id())) - 1
    )
    df_parse.write.parquet(out_path)

    print('end pyspark_processing job')
    end_pyspark_job = time.time()
    print("*** pyspark_processing job time: {:.2f}s ***".format(end_pyspark_job - start_pyspark_job))


if __name__ == '__main__':
    read_path = '/Users/iain/LocalStorage/TREC-NEWS/WashingtonPost.v2/data/TREC_Washington_Post_collection.v2.jl'
    dir_path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/test_news_chunks/'
    num_pages = 10
    chunks = 3
    print_intervals = 1
    write_output = True
    write_article_data_to_dir(read_path=read_path,
                              dir_path=dir_path,
                              num_pages=num_pages,
                              chunks=chunks,
                              print_intervals=print_intervals,
                              write_output=write_output)