
from pyspark.sql.types import BinaryType, BooleanType, StringType, ArrayType, FloatType
from pyspark.sql.functions import udf, row_number, monotonically_increasing_id, explode, desc
from pyspark.sql import SparkSession, Window
from utils.trec_car_tools import iter_pages, iter_paragraphs
import time
from protocol_buffers import document_pb2
import pickle

from pyspark_processing.pipeline import write_pages_data_to_dir, run_pyspark_pipeline


spark_drive_gbs = 80
spark_executor_gbs = 3
cores = 18

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
    entity_path = '/nfs/trec_car/data/test_entity/full_data_v3_with_datasets'
    out_path = '/nfs/trec_car/data/test_entity/full_data_v3_with_datasets_with_desc'
    df = spark.read.parquet(entity_path)

    @udf(returnType=StringType())
    def get_desc(doc_bytearray):
        doc = document_pb2.Document().FromString(pickle.loads(doc_bytearray))
        return '{}: {}'.format(doc.doc_id, doc.document_contents[0].text.split(".")[0])


    df_desc = df.withColumn("doc_bytearray", get_desc("doc_bytearray"))
    df_desc.write.parquet(out_path)

