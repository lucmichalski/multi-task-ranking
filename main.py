
import os

from metadata import CarEntityPaths, CarPassagePaths
from learning.experiments import FineTuningReRankingExperiments
from retrieval.dataset_processing import DatasetProcessing, BertTokenizer
from retrieval.tools import EvalTools, SearchTools, default_eval_config
from torch import nn

if __name__ == '__main__':

    # from metadata import NewsPassagePaths, CarEntityPaths
    # from retrieval.tools import SearchTools
    #
    # run_bases = ['/nfs/trec_news_track/runs/anserini/entity_2018/entity.custom_anserini.500000_doc.100_words.{}.fixed_qrels.run',
    #              '/nfs/trec_news_track/runs/anserini/entity_2019/entity.custom_anserini.500000_doc.100_words.{}.fixed_qrels.run',
    #              '/nfs/trec_news_track/runs/anserini/entity_2018/entity.custom_anserini.500000_doc.100_words.{}.fixed_qrels.run',
    #              '/nfs/trec_news_track/runs/anserini/entity_2019/entity.custom_anserini.500000_doc.100_words.{}.fixed_qrels.run']
    #
    # qrels_paths = ['/nfs/trec_news_track/data/2018/news_track.2018.entity.qrels',
    #                '/nfs/trec_news_track/data/2019/news_track.2019.entity.qrels',
    #                '/nfs/trec_news_track/data/2018/news_track.2018.entity.qrels',
    #                '/nfs/trec_news_track/data/2019/news_track.2019.entity.qrels'
    #                ]
    #
    # query_types = ['title+contents',
    #                'title+contents',
    #                'title',
    #                'title']
    # hits = 500000
    # news_index_path = NewsPassagePaths.index
    # words = 100
    #
    # for run_base, qrels_path, query_type in zip(run_bases, qrels_paths, query_types):
    #     run_path = run_base.format(query_types)
    #     search_tools = SearchTools(index_path=CarEntityPaths.index)
    #     search_tools.write_entity_run_news(run_path, qrels_path, query_type, words, hits, news_index_path)

    # from retrieval.dataset_processing import DatasetProcessing
    # from metadata import NewsPassagePaths
    #
    # qrels_path = '/nfs/trec_news_track/bert/train_entity/news_track.train.entity.qrels'
    # xml_topics_path = None #'/nfs/trec_news_track/data/2019/newsir19-background-linking-topics.xml'
    # run_path = '/nfs/trec_news_track/runs/anserini/bert/news_track.train.bm25.100000.title+contents.50_words.run'
    # index_path = NewsPassagePaths.index
    # car_index_path = CarEntityPaths.index
    # data_dir_path = '/nfs/trec_news_track/runs/anserini/bert/news_track_train_bm25_100000_50_words_bert_chunks/'
    # max_length = 512
    # context_path = None
    # training_dataset = True
    # ranking_type = 'entity'
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
    #                               car_index_path=car_index_path,
    #                               xml_topics_path=xml_topics_path)

    gpus = 1
    model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/test_trec_news_v2_entity_5e5_batch_8/epoch1_batch108/'
    extra_layers = False
    train_batch_size = 8 * gpus
    dev_batch_size = 64 * 3 * gpus

    # train_data_dir_path_passage =  None#'/nfs/trec_news_track/runs/anserini/bert/news_track_train_bm25_100000_50_words_bert_chunks/'
    # dev_data_dir_path_passage =  '/nfs/trec_news_track/runs/anserini/bert/news_track_dev_bm25_100000_50_words_bert_chunks/'
    # dev_qrels_path_passage = '/nfs/trec_news_track/bert/dev_entity/news_track.dev.entity.qrels'
    # dev_run_path_passage = '/nfs/trec_news_track/runs/anserini/bert/news_track.dev.bm25.100000.title+contents.50_words.run'

    train_data_dir_path_entity = None#'/nfs/trec_news_track/runs/anserini/bert/news_track_train_bm25_100000_50_words_bert_chunks/'
    dev_data_dir_path_entity = '/nfs/trec_news_track/runs/anserini/bert/news_track_test_bm25_100000_50_words_bert_chunks/'
    dev_qrels_path_entity = '/nfs/trec_news_track/bert/test_entity/news_track.test.entity.qrels'
    dev_run_path_entity = '/nfs/trec_news_track/runs/anserini/bert/news_track.test.bm25.100000.title+contents.50_words.run'

    experiment = FineTuningReRankingExperiments(model_path=model_path,
                                                extra_layers=extra_layers,
                                                train_batch_size=train_batch_size,
                                                dev_batch_size=dev_batch_size,

                                                train_data_dir_path_entity=train_data_dir_path_entity,
                                                dev_data_dir_path_entity=dev_data_dir_path_entity,
                                                dev_qrels_path_entity=dev_qrels_path_entity,
                                                dev_run_path_entity=dev_run_path_entity)
    #
    # epochs = 3
    # lr = 2e-5
    # eps = 1e-8
    # weight_decay = 0.01
    # warmup_percentage = 0.1
    # experiments_dir = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    # experiment_name = 'test_trec_news_v2_entity_5e5_batch_8'
    # write = True
    # logging_steps = 25
    # head_flag = 'entity'
    #
    # experiment.run_experiment_single_head(
    #     head_flag=head_flag,
    #     epochs=epochs,
    #     lr=lr,
    #     eps=eps,
    #     weight_decay=weight_decay,
    #     warmup_percentage=warmup_percentage,
    #     experiments_dir=experiments_dir,
    #     experiment_name=experiment_name,
    #     logging_steps=logging_steps
    # )

    head_flag = 'entity'
    rerank_run_path = '/nfs/trec_news_track/runs/bert/entity_2019/re_ranked_anserini.bm5.rm3.run'
    experiment.inference(head_flag=head_flag, rerank_run_path=rerank_run_path, do_eval=False, cap_rank=100)


