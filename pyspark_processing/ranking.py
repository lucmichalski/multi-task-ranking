
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

    df_with_links = df.withColumn("synthetic_entity_link_ids", get_synthetic_entity_link_ids("content_bytearray"))
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

##########################################################################
##########################################################################
##########################################################################
##########################################################################


def get_passage_id_map(xml_topics_path):
    """"""
    passage_id_map = {}
    with open(xml_topics_path, 'r') as f:
        for line in f:
            # Passage intermediate_id
            if '<num>' in line:
                start_i = [m.span() for m in re.finditer('<num> Number: ', line)][0][1]
                end_i = [m.span() for m in re.finditer(' </num>', line)][0][0]
                passage_temp_id = line[start_i:end_i]
            # Passage id
            if '<docid>' in line:
                start_i = [m.span() for m in re.finditer('<docid>', line)][0][1]
                end_i = [m.span() for m in re.finditer('</docid>', line)][0][0]
                passage_id = line[start_i:end_i]
                passage_id_map[passage_temp_id] = passage_id

    return passage_id_map


def get_top_100_rank(spark, run_path, rank_type='entity', k=100, xml_topics_path=None):
    """"""
    if rank_type == 'passage':
        print('using pasage id map')
        passage_id_map = get_passage_id_map(xml_topics_path=xml_topics_path)

    data = []
    with open(run_path, 'r', encoding='utf-8') as f_run:
        for line in f_run:
            query, _, doc_id, rank, _, _ = line.split()
            if int(rank) <= k:
                if rank_type == 'passage':
                    if query in passage_id_map:
                        data.append([passage_id_map[str(query)], str(doc_id), int(rank)])
                else:
                    data.append([str(query), str(doc_id), int(rank)])

    return spark.createDataFrame(data, ["query", "doc_id", "rank"])


@udf(returnType=ArrayType(StringType()))
def get_synthetic_entity_link_ids_passage(article_bytearray):
    """"""
    article = pickle.loads(article_bytearray)
    synthetic_entity_links = document_pb2.Document.FromString(article).document_contents[0].rel_entity_links
    return [str(s.entity_id) for s in synthetic_entity_links]


@udf(returnType=ArrayType(StringType()))
def get_synthetic_entity_link_ids_entity(doc_bytearray):
    """"""
    doc = pickle.loads(doc_bytearray)
    synthetic_entity_links = []
    for document_content in document_pb2.Document.FromString(doc).document_contents:
        synthetic_entity_links += document_content.synthetic_entity_links
    return [str(s.entity_id) for s in synthetic_entity_links]


def get_passage_df(spark, passage_run_path, xml_topics_path, passage_parquet_path):
    """"""
    passage_rank_df = get_top_100_rank(spark=spark,
                                       run_path=passage_run_path,
                                       rank_type='passage',
                                       k=100,
                                       xml_topics_path=xml_topics_path)
    passage_df = spark.read.parquet(passage_parquet_path)
    passage_df_with_entity_links = passage_df.withColumn("entity_links", get_synthetic_entity_link_ids_passage("article_bytearray"))
    passage_df_with_entity_links_reduced = passage_df_with_entity_links.select("doc_id", "entity_links").dropDuplicates()
    passage_join_df = passage_rank_df.join(passage_df_with_entity_links_reduced, on=['doc_id'], how='left')
    return passage_join_df


def get_entity_df(spark, entity_run_path, entity_parquet_path):
    """"""
    entity_rank_df = get_top_100_rank(spark=spark,
                                      run_path=entity_run_path,
                                      rank_type='entity',
                                      k=100,
                                      xml_topics_path=None)
    entity_df = spark.read.parquet(entity_parquet_path).select(col("page_id").alias("doc_id"), "doc_bytearray")
    entity_join_df = entity_rank_df.join(entity_df, on=['doc_id'], how='left')
    entity_df_with_entity_links = entity_join_df.withColumn("entity_links", get_synthetic_entity_link_ids_entity("doc_bytearray"))

    entity_df_with_entity_links_reduced = entity_df_with_entity_links.select("doc_id", "query", "rank", "entity_links").dropDuplicates()
    return entity_df_with_entity_links_reduced

