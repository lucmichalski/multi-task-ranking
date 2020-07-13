
import os

from metadata import CarEntityPaths, CarPassagePaths
from learning.experiments import FineTuningReRankingExperiments
from retrieval.dataset_processing import DatasetProcessing, BertTokenizer
from retrieval.tools import EvalTools, SearchTools, default_eval_config
from torch import nn

if __name__ == '__main__':

    # from retrieval.dataset_processing import DatasetProcessing
    # from metadata import NewsPassagePaths
    #
    # qrels_path = '/nfs/trec_news_track/data/2019/newsir19-qrels-background.txt'
    # xml_topics_path = '/nfs/trec_news_track/data/2019/newsir19-background-linking-topics.xml'
    # run_path = '/nfs/trec_news_track/bert/test_passage/news_track.test.passage.250.bm25.rm3.run'
    # index_path = NewsPassagePaths.index
    # data_dir_path = '/nfs/trec_news_track/bert/test_passage/news_track_test_passage_250_bm25_rm3_bert_chunks/'
    # max_length = 512
    # context_path = None
    # training_dataset = False
    # ranking_type = 'passage'
    # query_type = 'title+contents'
    #
    # processing = DatasetProcessing(qrels_path=qrels_path,
    #                                run_path=run_path,
    #                                index_path=index_path,
    #                                data_dir_path=data_dir_path,
    #                                max_length=max_length,
    #                                context_path=context_path)
    #
    # processing.build_news_dataset(training_dataset=training_dataset,
    #                               chuck_topic_size=1e8,
    #                               ranking_type=ranking_type,
    #                               query_type=query_type,
    #                               car_index_path=None,
    #                               xml_topics_path=xml_topics_path)

    gpus = 3
    model_path = None  # '/nfs/trec_car/data/bert_reranker_datasets/exp/bert_passages_with_top5_ents_6e6/epoch2_batch1500/'
    extra_layers = False
    train_batch_size = 8 * gpus
    dev_batch_size = 64 * 3 * gpus

    train_data_dir_path_passage = '/nfs/trec_news_track/bert/train_passage/news_track_train_passage_250_bm25_rm3_bert_chunks/'

    dev_data_dir_path_passage = '/nfs/trec_news_track/bert/dev_passage/news_track_dev_passage_250_bm25_rm3_bert_chunks/'

    dev_qrels_path_passage = '/nfs/trec_news_track/bert/dev_passage/news_track.dev.passage.qrels'

    dev_run_path_passage = '/nfs/trec_news_track/bert/dev_passage/news_track.dev.passage.250.bm25.rm3.run'

    experiment = FineTuningReRankingExperiments(model_path=model_path,
                                                extra_layers=extra_layers,
                                                train_batch_size=train_batch_size,
                                                dev_batch_size=dev_batch_size,

                                                train_data_dir_path_passage=train_data_dir_path_passage,

                                                dev_data_dir_path_passage=dev_data_dir_path_passage,

                                                dev_qrels_path_passage=dev_qrels_path_passage,

                                                dev_run_path_passage=dev_run_path_passage)

    epochs = 3
    lr = 2e-5
    eps = 1e-8
    weight_decay = 0.01
    warmup_percentage = 0.1
    experiments_dir = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    experiment_name = 'test_trec_news_v1'
    write = True
    logging_steps = 50

    experiment.run_experiment_multi_head(
        epochs=epochs,
        lr=lr,
        eps=eps,
        weight_decay=weight_decay,
        warmup_percentage=warmup_percentage,
        experiments_dir=experiments_dir,
        experiment_name=experiment_name,
        logging_steps=logging_steps
    )
