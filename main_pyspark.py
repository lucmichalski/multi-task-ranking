
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
    entity_path = '/nfs/trec_car/data/test_entity/full_data_v3_with_datasets_with_desc_v3/'
    out_path = '/nfs/trec_car/data/test_entity/full_data_v3_with_datasets_with_desc_v4_with_top5_ents_v2/'
    df = spark.read.parquet(entity_path)

    @udf(returnType=StringType())
    def get_desc(doc_bytearray):
        doc = document_pb2.Document().FromString(pickle.loads(doc_bytearray))
        try:
            return '{}: {}'.format(doc.doc_name, doc.document_contents[0].text.split(".")[0])
        except:
            return '{}: '.format(doc.doc_name)

    @udf(returnType=ArrayType(StringType()))
    def get_top_5_ents(doc_bytearray):
        synthetic_entity_link_totals = document_pb2.Document().FromString(pickle.loads(doc_bytearray)).synthetic_entity_link_totals

        link_counts = []
        for synthetic_entity_link_total in synthetic_entity_link_totals:
            entity_id = str(synthetic_entity_link_total.entity_id)
            count = sum([i.frequency for i in synthetic_entity_link_total.anchor_text_frequencies])
            link_counts.append((entity_id, count))

        return [i[0] for i in sorted(link_counts, key=lambda x: x[1], reverse=True)][:5]


    df_desc = df.withColumn("doc_desc", get_desc("doc_bytearray"))

    df_5_ent = df_desc.withColumn("top_5_ents", get_top_5_ents("doc_bytearray"))

    df_5_ent.write.parquet(out_path)

