
from pyspark.sql.types import BinaryType, BooleanType, StringType, ArrayType, FloatType
from pyspark.sql.functions import udf, row_number, monotonically_increasing_id, explode, desc, col, collect_list, concat_ws
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
    out_path = '/nfs/trec_car/data/test_entity/full_data_v3_with_datasets_with_desc_ents_context_v7/'

    df = spark.read.parquet(entity_path)

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
        synthetic_entity_link_totals = document_pb2.Document().FromString(pickle.loads(doc_bytearray)).synthetic_entity_link_totals
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


    # BUILD CONTEXT
    run_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity_synthetic_300.run'
    out_path = run_path + '.context.json'
    data_path = '/nfs/trec_car/data/test_entity/full_data_v3_with_datasets_with_desc_ents_context_v7/'

    df = spark.read.parquet(data_path)

    page_ids = []
    with open(run_path, 'r') as f:
        for i, line in enumerate(f):
            page_ids.append(line.split()[2])
    page_ids = [[i] for i in list(set(page_ids))]

    df_run = spark.createDataFrame(page_ids, ["page_id"])
    df_join = df_run.join(df, on=["page_id"], how='left').collect()
    entities_dict = {}

    for i in df_join:
        entities_dict[i[0]] = {
            'first_para': i[1],
            'top_ents': i[2]
        }
    import json
    with open(out_path, 'w') as fp:
        json.dump(entities_dict, fp, indent=4)


