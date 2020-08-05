
from pyspark.sql.types import BinaryType, BooleanType, StringType, ArrayType, FloatType
from pyspark.sql.functions import udf, row_number, monotonically_increasing_id, explode, desc, col, collect_list, concat_ws
from pyspark.sql import SparkSession, Window
from utils.trec_car_tools import iter_pages, iter_paragraphs
import time
from protocol_buffers import document_pb2
import pickle

from pyspark_processing.trec_car_pipeline import write_pages_data_to_dir, run_pyspark_pipeline


spark_drive_gbs = 50
spark_executor_gbs = 5
cores = 8

print('\n//////// RUNNING WITH CORES {} //////////'.format(cores))
spark = SparkSession.\
    builder\
    .appName('test')\
    .master('local[{}]'.format(cores)) \
    .config("spark.driver.memory", '{}g'.format(spark_drive_gbs)) \
    .config("spark.executor.memory", '{}g'.format(spark_executor_gbs)) \
    .config("spark.driver.maxResultSize", '{}g'.format(spark_drive_gbs)) \
    .getOrCreate()

if __name__ == '__main__':

    from document_parsing.trec_news_parsing import TrecNewsParser
    from pyspark_processing.trec_news_pipeline import write_article_data_to_dir, run_pyspark_pipeline

    read_path = '/nfs/trec_news_track/WashingtonPost.v2/data/TREC_Washington_Post_collection.v2.jl'
    dir_path = '/nfs/trec_news_track/index/test_10000_chunks/'
    num_docs = 100
    chunks = 10
    print_intervals = 10
    write_output = True
    rel_wiki_year = '2019'
    rel_base_url = '/nfs/trec_car/entity_processing/REL/'
    rel_model_path = rel_base_url + 'ed-wiki-{}/model'.format(rel_wiki_year)
    car_id_to_name_path = '/nfs/trec_news_track/lmdb.map_id_to_name.v1'

    # write_article_data_to_dir(read_path=read_path,
    #                           dir_path=dir_path,
    #                           num_docs=num_docs,
    #                           chunks=chunks,
    #                           print_intervals=print_intervals,
    #                           write_output=write_output)

    out_path = '/nfs/trec_news_track/index/test_10000_out_v4/'
    run_pyspark_pipeline(dir_path,
                         spark,
                         cores,
                         out_path,
                         rel_wiki_year,
                         rel_base_url,
                         rel_model_path,
                         car_id_to_name_path)

    # tnp = TrecNewsParser(rel_wiki_year, rel_base_url, rel_model_path, car_id_to_name_path)
    # tnp.parse_json_to_protobuf(read_path=read_path,
    #                            num_docs=num_docs,
    #                            write_output=write_output,
    #                            print_intervals=print_intervals)
