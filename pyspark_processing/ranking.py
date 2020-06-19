
from protocol_buffers import document_pb2

from pyspark.sql.types import StringType, ArrayType, FloatType
from pyspark.sql.functions import udf, row_number,  explode, desc

import pickle


def get_paragraph_data_from_run_file(spark, run_path, para_data, max_counter=10000000000):
    """ """
    # Create PySpark DF of run file - id, query, rank.
    content_ids = []
    with open(run_path, 'r') as f_run:
        counter = 0
        for line in f_run:
            query, _, content_id, rank, _, _ = line.split()
            counter += 1
            content_ids.append([str(content_id), str(query), int(rank)])
            if counter >= max_counter:
                break
    df_rank = spark.createDataFrame(content_ids, ["content_id", "query", "rank"])

    # Read and join to preprocessed paragraph data.
    df_para = spark.read.parquet(para_data)
    df = df_rank.join(df_para, on=['content_id'], how='left')

    return df

def get_ranked_entities_from_paragraph_data(df, output_path):
    """ """

    @udf(returnType=ArrayType(StringType()))
    def get_synthetic_entity_link_ids(content_bytearray):
        if content_bytearray == None:
            return []
        content = pickle.loads(content_bytearray)
        synthetic_entity_links = document_pb2.DocumentContent.FromString(content).synthetic_entity_links
        return [str(s.entity_id) for s in synthetic_entity_links]

    @udf(returnType=FloatType())
    def get_entity_links_count(synthetic_entity_link_ids):
        return float(len(synthetic_entity_link_ids))

    @udf(returnType=FloatType())
    def get_rank_weighting(rank):
        return 1.0/rank*rank

    @udf(returnType=FloatType())
    def get_norm_rank_weighting(rank_weighting, entity_links_count):
        if entity_links_count == 0.0:
            return 0.0
        return rank_weighting/entity_links_count

    df_with_links =  df.withColumn("synthetic_entity_link_ids", get_synthetic_entity_link_ids("content_bytearray"))
    df_with_links_with_counts = df_with_links.withColumn("entity_links_count", get_entity_links_count("synthetic_entity_link_ids"))

    df_with_links_exploded = df_with_links_with_counts.select("query", "rank", "entity_links_count", explode("synthetic_entity_link_ids").alias("synthetic_entity_link_id"))
    df_with_weighted_links_exploded =  df_with_links_exploded.withColumn("rank_weighting", get_rank_weighting("rank"))
    df_with_norm_weighted_links_exploded =  df_with_weighted_links_exploded.withColumn("norm_rank_weighting", get_norm_rank_weighting("rank_weighting", "entity_links_count"))

    df_entity_rank = df_with_norm_weighted_links_exploded.groupBy("query", "synthetic_entity_link_id").agg({"norm_rank_weighting": "sum"})

    query_df = df_entity_rank.sort("query", desc("sum(norm_rank_weighting)")).collect()

    with open(output_path, 'w') as f:
        old_query = ''
        for query, synthetic_entity_link_id, rank_weight in query_df:
            if old_query != query:
                rank = 1
            f.write(" ".join((query, "Q0", str(synthetic_entity_link_id), str(rank), "{:.6f}".format(rank_weight), "ENTITY-LINKS")) + '\n')
            old_query = query
            rank += 1