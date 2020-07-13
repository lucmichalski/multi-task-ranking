
import os

from metadata import CarEntityPaths, CarPassagePaths
from learning.experiments import FineTuningReRankingExperiments
from retrieval.dataset_processing import DatasetProcessing, BertTokenizer
from retrieval.tools import EvalTools, SearchTools, default_eval_config
from torch import nn

if __name__ == '__main__':

    from retrieval.dataset_processing import DatasetProcessing
    from metadata import NewsPassagePaths

    qrels_path = '/nfs/trec_news_track/bert/dev_passage/news_track.dev.passage.qrels'
    xml_topics_path = '/nfs/trec_news_track/data/2018/newsir18-topics.txt'
    run_path = '/nfs/trec_news_track/bert/dev_passage/news_track.dev.passage.250.bm25.rm3.run'
    index_path = NewsPassagePaths.index
    data_dir_path = '/nfs/trec_news_track/bert/dev_passage/news_track_dev_passage_250_bm25_rm3_bert_chunks/'
    max_length = 512
    context_path = None

    processing = DatasetProcessing(qrels_path=qrels_path,
                                   run_path=run_path,
                                   index_path=index_path,
                                   data_dir_path=data_dir_path,
                                   max_length=max_length,
                                   context_path=context_path)

    processing.build_news_dataset(training_dataset=False,
                                  chuck_topic_size=1e8,
                                  ranking_type='passage',
                                  query_type='title+contents',
                                  car_index_path=None,
                                  xml_topics_path=xml_topics_path)