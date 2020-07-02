
from pyspark.sql.types import BinaryType, BooleanType, StringType, ArrayType, FloatType
from pyspark.sql.functions import udf, row_number, monotonically_increasing_id, explode, desc
from pyspark.sql import SparkSession, Window
from utils.trec_car_tools import iter_pages, iter_paragraphs
import time
from protocol_buffers import document_pb2
import pickle

from pyspark_processing.pipeline import write_pages_data_to_dir, run_pyspark_pipeline


spark_drive_gbs = 50
spark_executor_gbs = 2
cores = 14

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
    read_paths = ['/home/iain_mackie1993/nfs/data/pages_corpus/full_unprocessedAllButBenchmark/unprocessedAllButBenchmark.Y2.cbor',
                  '/home/iain_mackie1993/nfs/data/trec_car/multi-rank-data-gcp/benchmarkY1/benchmarkY1-test/test.pages.cbor ',
                  '/home/iain_mackie1993/nfs/data/trec_car/multi-rank-data-gcp/benchmarkY1/benchmarkY1-train/train.pages.cbor',
                  '/home/iain_mackie1993/nfs/data/trec_car/multi-rank-data-gcp/test200/test200-train/train.pages.cbor',
                  '/home/iain_mackie1993/nfs/data/trec_car/multi-rank-data-gcp/benchmarkY2test/benchmarkY2test-goldarticles.cbor ']
    datasets = ['unprocessedAllButBenchmark',
                'benchmarkY1-test',
                'benchmarkY1-train',
                'test200',
                'benchmarkY2-test']
    chunks = 100000
    dir_path = '/home/iain_mackie1993/nfs/data/trec_car/entity_links/pages_chunks/'
    for read_path, dataset in zip(read_paths, datasets):

        write_pages_data_to_dir(read_path,
                                dir_path=dir_path,
                                num_pages=100000000000,
                                dataset=dataset,
                                chunks=chunks,
                                print_intervals=100,
                                write_output=True)

    out_path = '/home/iain_mackie1993/nfs/data/trec_car/entity_links/pyspark_processed_pages_v1/'
    run_pyspark_pipeline(dir_path, spark, cores, out_path)