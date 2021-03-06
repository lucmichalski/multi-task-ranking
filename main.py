
import os

from metadata import CarEntityPaths, CarPassagePaths, NewsPassagePaths
from learning.experiments import FineTuningReRankingExperiments
from retrieval.dataset_processing import DatasetProcessing, BertTokenizer
from retrieval.tools import EvalTools, SearchTools, default_eval_config
from torch import nn

from multi_task.processing import MultiTaskDataset, create_extra_queries, MultiTaskDatasetByQuery
from multi_task.ranking import train_mutant_multi_task_max_combo_news, train_mutant_multi_task_max_combo


if __name__ == '__main__':

    # dir_path = '/nfs/trec_news_track/data/5_fold/'
    # passage_model_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/exp/passage_ranking_bert_train_2epoch+8e-06lr/epoch1_batch2000/'
    # entity_model_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/exp/entity_ranking_bert_train_2epoch+8e-06lr/epoch2_batch6000/'
    # batch_size = 64*2
    # max_rank = 100
    # query_keyword_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/news_tf_idf_queries_no_stem_107_queries.json'
    # doc_keyword_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/news_tf_idf_docs_no_stem.json'
    # MultiTaskDatasetByQuery().build_dataset_by_query_entity_context_news_keyword(dir_path=dir_path,
    #                                                                              max_rank=max_rank,
    #                                                                              batch_size=batch_size,
    #                                                                              passage_model_path=passage_model_path,
    #                                                                              entity_model_path=entity_model_path,
    #                                                                              query_keyword_path=query_keyword_path,
    #                                                                              doc_keyword_path=doc_keyword_path)
    for batch_size in [256]:
        for lr in [0.0005, 0.0001, 0.00001]:
            train_mutant_multi_task_max_combo(batch_size=batch_size, lr=lr, mutant_type='mean')
            train_mutant_multi_task_max_combo(batch_size=batch_size, lr=lr, mutant_type='max')



    # dir_path = '/nfs/trec_news_track/data/5_fold/'
    # passage_model_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/exp/passage_ranking_bert_train_2epoch+8e-06lr/epoch1_batch2000/'
    # entity_model_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/exp/entity_ranking_bert_train_2epoch+8e-06lr/epoch2_batch6000/'
    # batch_size = 64*2
    # max_rank = 100
    # MultiTaskDatasetByQuery().build_dataset_by_query_entity_context_news(dir_path=dir_path,
    #                                                                     max_rank=max_rank,
    #                                                                     batch_size=batch_size,
    #                                                                     passage_model_path=passage_model_path,
    #                                                                     entity_model_path=entity_model_path
    #                                                                     )

    # from multi_task.ranking import train_cls_model_max_combo, train_cls_model, train_mutant_max_combo
    #
    # batch_sizes = [64, 256]
    # lrs = [0.00001, 0.0001, 0.0005, 0.001]
    # for batch_size in batch_sizes:
    #     for lr in lrs:
    # #         train_cls_model(batch_size=batch_size, lr=lr)
    #         train_mutant_max_combo(batch_size=batch_size, lr=lr)

    # dir_path = '/nfs/trec_car/data/entity_ranking/multi_task_data_by_query_1000/'
    # passage_model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/benchmarkY1_passage_100_lr_8e6_num_warmup_steps_1000/epoch1_batch14000/'
    # entity_model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/full_data_v2_hierarchical_10000_hits_300_v2_lr_2e6_num_warmup_steps_0.1_new_pipeline/epoch1_batch420000/'
    # batch_size = 64*6
    # max_rank = 1000
    # MultiTaskDatasetByQuery().build_dataset_by_query_entity_context(dir_path=dir_path,
    #                                                                 max_rank=max_rank,
    #                                                                 batch_size=batch_size,
    #                                                                 passage_model_path=passage_model_path,
    #                                                                 entity_model_path=entity_model_path
    #                                                                 )
    # from multi_task.ranking import rerank_runs, train_cls_model_max_combo, train_cls_model
    #
    # dir_path = '/nfs/trec_car/data/entity_ranking/multi_task_data_by_query/'
    # passage_model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/benchmarkY1_passage_100_lr_8e6_num_warmup_steps_1000/epoch1_batch14000/'
    # entity_model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/full_data_v2_hierarchical_10000_hits_300_v2_lr_2e6_num_warmup_steps_0.1_new_pipeline/epoch1_batch420000/'
    # MultiTaskDatasetByQuery().build_dataset_by_query(dir_path=dir_path,
    #                                                  max_rank=100,
    #                                                  batch_size=64,
    #                                                  bi_encode=True,
    #                                                  passage_model_path=passage_model_path,
    #                                                  entity_model_path=entity_model_path
    #                                                  )

    # dataset = 'test'
    # batch_sizes = [256, 1024]
    # lrs = [0.00001, 0.0001, 0.0005, 0.001]
    # for batch_size in batch_sizes:
    #     for lr in lrs:
    #         train_cls_model(batch_size=batch_size, lr=lr)

    # dir_path = '/nfs/trec_car/data/entity_ranking/multi_task_data_by_query_1000/'
    # passage_model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/benchmarkY1_passage_100_lr_8e6_num_warmup_steps_1000/epoch1_batch14000/'
    # entity_model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/full_data_v2_hierarchical_10000_hits_300_v2_lr_2e6_num_warmup_steps_0.1_new_pipeline/epoch1_batch420000/'
    # MultiTaskDatasetByQuery().build_dataset_by_query(dir_path=dir_path,
    #                                                  max_rank=1000,
    #                                                  batch_size=64,
    #                                                  passage_model_path=passage_model_path,
    #                                                  entity_model_path=entity_model_path
    #                                                  )



    # =====================================================
    # ================ TREC CAR PROCESSING=================
    # =====================================================
    # qrels_paths = ['/nfs/trec_car/data/entity_ranking/testY2_automatic_entity_data/testY2_automatic_entity.qrels',
    #                '/nfs/trec_car/data/entity_ranking/testY2_manual_entity_data/testY2_manual_entity.qrels']
    # run_paths = ['/nfs/trec_car/data/entity_ranking/testY2_automatic_entity_data/testY2_automatic_entity_1000.run',
    #              '/nfs/trec_car/data/entity_ranking/testY2_manual_entity_data/testY2_manual_entity_1000.run']
    # data_dir_paths = ['/nfs/trec_car/data/entity_ranking/testY2_automatic_entity_data/testY2_automatic_entity_1000_chunks/',
    #                   '/nfs/trec_car/data/entity_ranking/testY2_manual_entity_data/testY2_manual_entity_1000_chunks/']
    #
    # for qrels_path, run_path, data_dir_path in zip(qrels_paths, run_paths, data_dir_paths):
    #     index_path = CarEntityPaths.index
    #     max_length = 512
    #     context_path = None
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #     binary_qrels = True
    #     dp = DatasetProcessing(qrels_path=qrels_path,
    #                            run_path=run_path,
    #                            index_path=index_path,
    #                            data_dir_path=data_dir_path,
    #                            max_length=max_length,
    #                            context_path=context_path,
    #                            tokenizer=tokenizer,
    #                            binary_qrels=binary_qrels)
    #
    #     chuck_topic_size = 100
    #     dp.build_car_dataset(training_dataset=False, chuck_topic_size=100, first_para=True)

    # =====================================================
    # ==================== TREC CAR =======================
    # =====================================================
    # gpus = 2
    # model_path = None
    # dev_batch_size = 64 * 4 * gpus
    # train_batch_size = 8 * gpus
    #
    # train_data_dir_path_entity = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_train_data/benchmarkY1_train_entity_synthetic_300_chunks/'
    # dev_data_dir_path_entity = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity_synthetic_300_chunks/'
    # dev_qrels_path_entity = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity_synthetic.qrels'
    # dev_run_path_entity = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity_synthetic_300.run'
    #
    # train_data_dir_path_passage = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_passage_train_data/benchmarkY1_hierarchical_passage_train_100_chunks/'
    # dev_data_dir_path_passage = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_passage_dev_data/benchmarkY1_hierarchical_passage_dev_100_chunks/'
    # dev_qrels_path_passage = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_passage_dev_data/benchmarkY1_dev_passage.qrels'
    # dev_run_path_passage = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_passage_dev_data/benchmarkY1_dev_passage_100.run'
    #
    # experiments = FineTuningReRankingExperiments(
    #     model_path = None,
    #     extra_layers = False,
    #     dev_batch_size = dev_batch_size,
    #     train_batch_size = train_batch_size,
    #
    #     train_data_dir_path_entity = train_data_dir_path_entity,
    #     dev_data_dir_path_entity = dev_data_dir_path_entity,
    #     dev_qrels_path_entity = dev_qrels_path_entity,
    #     dev_run_path_entity = dev_run_path_entity,
    #
    #     train_data_dir_path_passage = train_data_dir_path_passage,
    #     dev_data_dir_path_passage = dev_data_dir_path_passage,
    #     dev_qrels_path_passage = dev_qrels_path_passage,
    #     dev_run_path_passage = dev_run_path_passage,
    # )
    #
    # epochs = 2
    # lr = 1e-5
    # experiments_dir = '/nfs/trec_car/data/entity_ranking/testY1_hierarchical_passage_data/exp/'
    # experiment_name = 'multi_task_ranking_bert_no_sigmoid_{}epoch+{}lr'.format(epochs, lr)
    # experiments.run_experiment_multi_head(
    #     epochs=epochs,
    #     lr=lr,
    #     eps=1e-8,
    #     weight_decay=0.01,
    #     warmup_percentage=0.1,
    #     experiments_dir=experiments_dir,
    #     experiment_name=experiment_name,
    #     logging_steps=500)

    # =====================================================
    # =====================================================
    # =====================================================

    # dir_path = '/nfs/trec_car/data/entity_ranking/multi_task_data_by_query_1000/'
    # MultiTaskDatasetByQuery().build_dataset_by_query(dir_path=dir_path, max_rank=1000)

    #dir_path = '/nfs/trec_car/data/entity_ranking/multi_task_data_new/'
    #
    # MultiTaskDataset().build_datasets(dir_path=dir_path)
    #
    #MultiTaskDataset().cls_processing(dir_path=dir_path, batch_size=64*10)

    #create_extra_queries(dir_path='/nfs/trec_car/data/entity_ranking/multi_task_data/')

    #MultiTaskDataset().cls_processing(dir_path=dir_path, batch_size=64*10)

    # query_type = 'title+contents'
    # words = 100
    # folds = [0,1,2,3,4]
    # models = ['bm25', 'bm25_rm3']
    # datasets = ['test', 'valid', 'train']
    # hits_list = [500, 500, 1000]
    # index_path = CarEntityPaths.index
    #
    # for fold in folds:
    #     for model in models:
    #         base_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/'.format(fold)
    #         run_paths = [base_path + 'passage_{}_{}.run'.format(i, model) for i in datasets]
    #         qrels_paths = [base_path + 'passage_{}.qrels'.format(i) for i in datasets]
    #         for run_path, qrels_path, hits in zip(run_paths, qrels_paths, hits_list):
    #             if model == 'bm25':
    #                 searcher_config = {
    #                     'BM25': {'k1': 0.9,
    #                              'b': 0.4}
    #                 }
    #             else:
    #                 searcher_config = {
    #                     'BM25+RM3': {'BM25':
    #                                      {'k1': 0.9,
    #                                       'b': 0.4},
    #                                  'RM3':
    #                                      {'fb_terms': 10,
    #                                       'fb_docs': 10,
    #                                       'original_query_weight': 0.5}
    #                                  }
    #                 }
    #
    #             search_tools = SearchTools(index_path=index_path, searcher_config=searcher_config)
    #
    #             search_tools.write_entity_run_news(run_path=run_path,
    #                                                qrels_path=qrels_path,
    #                                                query_type=query_type,
    #                                                words=words,
    #                                                hits=hits,
    #                                                news_index_path=NewsPassagePaths.index)
    # datasets = ['test', 'valid', 'train']
    # folds = [0,1,2,3,4]
    # training_datasets = [False, False, True]
    # for fold in folds:
    #     for dataset, training_dataset in zip(datasets, training_datasets):
    #         qrels_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/passage_{}.qrels'.format(fold, dataset)
    #         run_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/passage_{}_ranking_1000.run'.format(fold, dataset)
    #         index_path = NewsPassagePaths.index
    #         data_dir_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/passage_{}_bert_ranking_data_keyword_v2/'.format(fold, dataset)
    #         max_length = 512
    #         context_path = None
    #         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #         binary_qrels = True
    #         keyword_dict_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/news_tf_idf_queries_no_stem_107_queries.json'
    #         keyword_passage_dict_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/news_tf_idf_docs_no_stem.json'
    #         dp = DatasetProcessing(qrels_path=qrels_path,
    #                                run_path=run_path,
    #                                index_path=index_path,
    #                                data_dir_path=data_dir_path,
    #                                max_length=max_length,
    #                                context_path=context_path,
    #                                tokenizer=tokenizer,
    #                                binary_qrels=binary_qrels)
    #
    #         chuck_topic_size = 100
    #         ranking_type = 'passage'
    #         query_type = 'title+contents'
    #         car_index_path = CarEntityPaths.index
    #         dp.build_news_dataset(training_dataset=training_dataset,
    #                               chuck_topic_size=chuck_topic_size,
    #                               ranking_type=ranking_type,
    #                               query_type=query_type,
    #                               car_index_path=car_index_path,
    #                               keyword_dict_path=keyword_dict_path,
    #                               keyword_passage_dict_path=keyword_passage_dict_path)
    #
    # datasets = ['test', 'valid', 'train']
    # folds = [0,1,2,3,4]
    # training_datasets = [False, False, True]
    # for fold in folds:
    #     for dataset, training_dataset in zip(datasets, training_datasets):
    #         qrels_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/entity_{}.qrels'.format(fold, dataset)
    #         run_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/entity_{}_BM25_ranking_1000.run'.format(fold, dataset)
    #         index_path = NewsPassagePaths.index
    #         data_dir_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/entity_{}_bert_ranking_data_keyword_v2/'.format(fold, dataset)
    #         max_length = 512
    #         context_path = None
    #         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #         binary_qrels = True
    #         keyword_dict_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/news_tf_idf_queries_no_stem_107_queries.json'
    #         dp = DatasetProcessing(qrels_path=qrels_path,
    #                                run_path=run_path,
    #                                index_path=index_path,
    #                                data_dir_path=data_dir_path,
    #                                max_length=max_length,
    #                                context_path=context_path,
    #                                tokenizer=tokenizer,
    #                                binary_qrels=binary_qrels)
    #
    #         chuck_topic_size = 100
    #         ranking_type = 'entity'
    #         query_type = 'title+contents'
    #         car_index_path = CarEntityPaths.index
    #         dp.build_news_dataset(training_dataset=training_dataset,
    #                               chuck_topic_size=chuck_topic_size,
    #                               ranking_type=ranking_type,
    #                               query_type=query_type,
    #                               car_index_path=car_index_path,
    #                               keyword_dict_path=keyword_dict_path)
    #1
    # lr = 9e-6
    # epochs = 2
    # gpus = 2
    # passage_model_paths = [
    #     '/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/exp/passage_ranking_keyword_query_bert_train_2epoch+2e-05lr/epoch1_batch1000/',
    #     '/nfs/trec_news_track/data/5_fold/scaled_5fold_1_data/exp/passage_ranking_keyword_query_bert_train_2epoch+2e-05lr/epoch1_batch1000/',
    #     '/nfs/trec_news_track/data/5_fold/scaled_5fold_2_data/exp/passage_ranking_keyword_query_bert_train_2epoch+2e-05lr/epoch1_batch1000/',
    #     '/nfs/trec_news_track/data/5_fold/scaled_5fold_3_data/exp/passage_ranking_keyword_query_bert_train_2epoch+2e-05lr/epoch1_batch1000/',
    #     '/nfs/trec_news_track/data/5_fold/scaled_5fold_4_data/exp/passage_ranking_keyword_query_bert_train_2epoch+2e-05lr/epoch1_batch1000/',
    # ]
    # entity_model_paths = [
    #     '/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/exp/entity_ranking_keyword_query_bert_train_2epoch+2e-05lr/epoch1_batch2000/',
    #     '/nfs/trec_news_track/data/5_fold/scaled_5fold_1_data/exp/entity_ranking_keyword_query_bert_train_2epoch+2e-05lr/epoch1_batch2000/',
    #     '/nfs/trec_news_track/data/5_fold/scaled_5fold_2_data/exp/entity_ranking_keyword_query_bert_train_2epoch+2e-05lr/epoch1_batch2000/',
    #     '/nfs/trec_news_track/data/5_fold/scaled_5fold_3_data/exp/entity_ranking_keyword_query_bert_train_2epoch+2e-05lr/epoch1_batch2000/',
    #     '/nfs/trec_news_track/data/5_fold/scaled_5fold_4_data/exp/entity_ranking_keyword_query_bert_train_2epoch+2e-05lr/epoch1_batch2000/',
    # ]
    # for task in ['entity', 'passage']:
    #     if task == 'passage':
    #         model_paths = passage_model_paths
    #     else:
    #         model_paths = entity_model_paths
    #     folds = [0, 1, 2, 3, 4]
    #     for fold in folds:
    #         dev_batch_size = 64 * 8 * gpus
    #         train_batch_size = 8 * gpus
    #         experiments_dir = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/exp/'.format(fold)
    #         experiment_name = '{}_keyword_v2_{}_bert'.format(task, lr)
    #
    #         if task == 'passage':
    #             train_data_dir_path_passage = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/passage_train_bert_ranking_data_keyword_v2/'.format(fold)
    #             dev_data_dir_path_passage = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/passage_valid_bert_ranking_data_keyword_v2/'.format(fold)
    #             dev_qrels_path_passage = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/passage_valid.qrels'.format(fold)
    #             dev_run_path_passage= '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/passage_valid_ranking_1000.run'.format(fold)
    #
    #             # dev_data_dir_path_passage = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/passage_test_bert_ranking_data_keyword/'.format(fold)
    #             # dev_qrels_path_passage = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/passage_test.qrels'.format(fold)
    #             # dev_run_path_passage= '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/passage_test_ranking_1000.run'.format(fold)
    #
    #             experiments = FineTuningReRankingExperiments(
    #                 model_path=None,
    #                 extra_layers=False,
    #                 train_batch_size=train_batch_size,
    #                 dev_batch_size=dev_batch_size,
    #
    #                 train_data_dir_path_passage=train_data_dir_path_passage,
    #                 dev_data_dir_path_passage=dev_data_dir_path_passage,
    #                 dev_qrels_path_passage=dev_qrels_path_passage,
    #                 dev_run_path_passage=dev_run_path_passage,
    #             )
    #
    #         else:
    #             train_data_dir_path_entity = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/entity_train_bert_ranking_data_keyword_v2/'.format(fold)
    #             dev_data_dir_path_entity = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/entity_valid_bert_ranking_data_keyword_v2/'.format(fold)
    #             dev_qrels_path_entity = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/entity_valid.qrels'.format(fold)
    #             dev_run_path_entity = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/entity_valid_BM25_ranking_1000.run'.format(fold)
    #             # dev_data_dir_path_entity = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/entity_test_bert_ranking_data_keyword/'.format(fold)
    #             # dev_qrels_path_entity = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/entity_test.qrels'.format(fold)
    #             # dev_run_path_entity = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/entity_test_BM25_ranking_1000.run'.format(fold)
    #
    #             experiments = FineTuningReRankingExperiments(
    #                 model_path=None,
    #                 extra_layers=False,
    #                 train_batch_size=train_batch_size,
    #                 dev_batch_size=dev_batch_size,
    #
    #                 train_data_dir_path_entity=train_data_dir_path_entity,
    #                 dev_data_dir_path_entity=dev_data_dir_path_entity,
    #                 dev_qrels_path_entity=dev_qrels_path_entity,
    #                 dev_run_path_entity=dev_run_path_entity,
    #             )
    #
    #         experiments.run_experiment_single_head(
    #             head_flag=task,
    #             epochs=epochs,
    #             lr=lr,
    #             eps=1e-8,
    #             weight_decay=0.01,
    #             warmup_percentage=0.1,
    #             experiments_dir=experiments_dir,
    #             experiment_name=experiment_name,
    #             logging_steps=500)
    #         # rerank_run_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_{}_data/{}_keyword_bert_ranking_1000.run'.format(fold, task)
    #         # experiments.inference(head_flag=task, rerank_run_path=rerank_run_path, cap_rank=100, do_eval=False)

    # # hits = 1000
    # # printing_step = 50
    # # run_path = '/nfs/trec_car/data/entity_ranking/testY2_automatic_entity_data/testY2_automatic_entity_bm25_rm3_1000.run.v2'
    # # topics_path = '/nfs/trec_car/data/entity_ranking/testY2_automatic_entity_data/testY2_automatic_entity.topics'
    #run_path = '/nfs/trec_car/data/entity_ranking/testY2_manual_entity_data/testY2_manual_entity_bm25_rm3_1000.run.v2'
    #topics_path = '/nfs/trec_car/data/entity_ranking/testY2_manual_entity_data/testY2_manual_entity.topics'
    # search_tools.write_run_car(topics_path, run_path, hits=hits, printing_step=printing_step)

    # ev = EvalTools()
    # run_path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/just_graph.run'
    # qrels_path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/TREC-NEWS/2018/news_track.2018.passage.qrels'
    # eval_path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/just_graph.run.eval.v1'
    # ev.write_eval_from_qrels_and_run(run_path=run_path,
    #                                  qrels_path=qrels_path,
    #                                  eval_path=eval_path)
    # alphas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    # betas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    #
    # for alpha in alphas:
    #     for beta in betas:
    #         s = """anserini/eval/trec_eval.9.0.4/trec_eval -c -M1000 -m ndcg_cut.5 -c -M1000 -m map -c -M1000 -m recip_rank -c -M1000 -m P.20 -c -M1000 -m ndcg_cut.20 -c -M1000 -m Rprec -c -M1000 -m recall.40 -c -M1000 -m recall.100 /nfs/trec_news_track/data/2018/news_track.2018.passage.qrels /nfs/trec_news_track/runs/anserini/graph/norm_combined_entity_graph_scores_alpha_{}_beta_{}.run > /nfs/trec_news_track/runs/anserini/graph/combined_entity_graph_scores_alpha_{}_beta_{}.run.eval.v1""".format(alpha, beta, alpha, beta)
    #         print(s)
    #         print("")

    from retrieval.tools import RetrievalUtils
    import random






    # from document_parsing.trec_news_parsing import TrecNewsParser
    #
    # path = '/Users/iain/LocalStorage/TREC-NEWS/WashingtonPost.v2/data/TREC_Washington_Post_collection.v2.jl'
    # rel_base_url = "/Users/iain/LocalStorage/coding/github/REL/"
    # rel_wiki_year = '2019'
    # rel_model_path = "/Users/iain/LocalStorage/coding/github/REL/ed-wiki-{}/model".format(rel_wiki_year)
    # car_id_to_name_path = '/Users/iain/LocalStorage/lmdb.map_id_to_name.v1'
    # print_intervals = 100
    # num_docs = 50000
    # chunks = 500
    # write_output = True
    # dir_path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/2018_bm25_rm3_chunks_full_v1/'
    # tnp = TrecNewsParser(rel_wiki_year=rel_wiki_year,
    #                      rel_base_url=rel_base_url,
    #                      rel_model_path=rel_model_path,
    #                      car_id_to_name_path=car_id_to_name_path)
    #
    # index_path = '/Users/iain/LocalStorage/TREC-NEWS/lucene-index-copy'
    # run_path = '/Users/iain/Downloads/anserini.bm5.rm3.default (1).run'
    # tnp.parse_run_file_to_parquet(run_path=run_path,
    #                               index_path=index_path,
    #                               write_output=write_output,
    #                               dir_path=dir_path,
    #                               num_docs=num_docs,
    #                               chunks=chunks,
    #                               print_intervals=print_intervals)

    # from metadata import NewsPassagePaths, CarEntityPaths
    # from retrieval.tools import SearchTools

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

    # run_paths = ['/nfs/trec_news_track/runs/anserini/bert/news_track.dev.bm25.100000.title+contents.50_words.run',
    #              '/nfs/trec_news_track/runs/anserini/bert/news_track.test.bm25.100000.title+contents.50_words.run']
    # qrels_paths = ['/nfs/trec_news_track/bert/dev_entity/news_track.dev.entity.qrels',
    #                '/nfs/trec_news_track/bert/test_entity/news_track.test.entity.qrels']
    # query_type = 'title+contents'
    #
    # hits = 100000
    # news_index_path = NewsPassagePaths.index
    # words = 50
    #
    # for run_path, qrels_path in zip(run_paths, qrels_paths):
    #     search_tools = SearchTools(index_path=CarEntityPaths.index)
    #     search_tools.write_entity_run_news(run_path, qrels_path, query_type, words, hits, news_index_path)

    # from retrieval.dataset_processing import DatasetProcessing
    # from metadata import NewsPassagePaths
    #
    #TEST
    # qrels_path = '/nfs/trec_news_track/bert/test_passage/news_track.test.passage.qrels'
    # xml_topics_path = '/nfs/trec_news_track/data/2019/newsir19-background-linking-topics.xml'
    # run_path = '/nfs/trec_news_track/bert/test_passage/news_track.test.passage.250.bm25.rm3.run'
    # index_path = NewsPassagePaths.index
    # car_index_path = None #CarEntityPaths.index
    # data_dir_path = '/nfs/trec_news_track/bert/test_passage/news_track_test_passage_250_bm25_rm3_bert_chunks_scaled_rel/'
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
    #                               car_index_path=car_index_path,
    #                               xml_topics_path=xml_topics_path)
    #
    # gpus = 2
    # model_paths = ['/nfs/trec_car/data/entity_ranking/testY1_hierarchical_passage_data/exp/multi_task_ranking_bert_2epoch+3e-06lr/epoch2_batch1000/',
    #                '/nfs/trec_car/data/entity_ranking/testY1_hierarchical_passage_data/exp/multi_task_ranking_bert_2epoch+5e-06lr/epoch1_batch1000/',
    #                '/nfs/trec_car/data/entity_ranking/testY1_hierarchical_passage_data/exp/multi_task_ranking_bert_2epoch+5e-06lr/epoch1_batch1000/']
    # head_flags = ['passage', 'entity', 'entity']
    # names = ['passage_hier_Y1_test.run', 'entity_auto_Y2_test.run', 'entity_manual_Y2_test.run']
    # model_paths = ['/nfs/trec_car/data/entity_ranking/testY1_hierarchical_passage_data/exp/multi_task_ranking_bert_no_sigmoid_2epoch+8e-06lr/epoch1_batch6000/',
    #                '/nfs/trec_car/data/entity_ranking/testY1_hierarchical_passage_data/exp/multi_task_ranking_bert_no_sigmoid_2epoch+1e-05lr/epoch1_batch3500/',
    #                '/nfs/trec_car/data/entity_ranking/testY1_hierarchical_passage_data/exp/multi_task_ranking_bert_no_sigmoid_2epoch+1e-05lr/epoch1_batch3500/']
    # head_flags = ['passage', 'entity', 'entity']
    # names = ['multi_task_passage_Y1_test_no_sigmoid.run', 'multi_task_entity_auto_Y2_test_no_sigmoid.run', 'multi_task_entity_manual_Y2_test_no_sigmoid.run']
    # for model_path, head_flag, name in zip(model_paths, head_flags, names):
    #     extra_layers = False
    #     train_batch_size = None #8 * gpus
    #     dev_batch_size = 64 * 8 * gpus
    #
    #     train_data_dir_path_passage = None #'/nfs/trec_news_track/bert/train_passage/news_track_train_passage_250_bm25_rm3_bert_chunks_scaled_rel/'
    #     dev_data_dir_path_passage =  '/nfs/trec_car/data/entity_ranking/testY1_hierarchical_passage_data/testY1_hierarchical_passage_1000_chunks/'
    #     dev_qrels_path_passage = '/nfs/trec_car/data/entity_ranking/testY1_hierarchical_passage_data/testY1_hierarchical_passage.qrels'
    #     dev_run_path_passage = '/nfs/trec_car/data/entity_ranking/testY1_hierarchical_passage_data/testY1_hierarchical_passage_1000.run'
    #
    #     if 'auto' in name:
    #         train_data_dir_path_entity = None #'/nfs/trec_news_track/runs/anserini/bert/news_track_train_bm25_100000_50_words_bert_chunks_scaled_rel/'
    #         dev_data_dir_path_entity = '/nfs/trec_car/data/entity_ranking/testY2_automatic_entity_data/testY2_automatic_entity_1000_chunks/'
    #         dev_qrels_path_entity = '/nfs/trec_car/data/entity_ranking/testY2_automatic_entity_data/testY2_automatic_entity.qrels'
    #         dev_run_path_entity = '/nfs/trec_car/data/entity_ranking/testY2_automatic_entity_data/testY2_automatic_entity_1000.run'
    #     else:
    #         train_data_dir_path_entity = None  # '/nfs/trec_news_track/runs/anserini/bert/news_track_train_bm25_100000_50_words_bert_chunks_scaled_rel/'
    #         dev_data_dir_path_entity = '/nfs/trec_car/data/entity_ranking/testY2_manual_entity_data/testY2_manual_entity_1000_chunks/'
    #         dev_qrels_path_entity = '/nfs/trec_car/data/entity_ranking/testY2_manual_entity_data/testY2_manual_entity.qrels'
    #         dev_run_path_entity = '/nfs/trec_car/data/entity_ranking/testY2_manual_entity_data/testY2_manual_entity_1000.run'
    #
    #     if head_flag == 'passage':
    #         experiment = FineTuningReRankingExperiments(model_path=model_path,
    #                                                     extra_layers=extra_layers,
    #                                                     train_batch_size=train_batch_size,
    #                                                     dev_batch_size=dev_batch_size,
    #
    #                                                     train_data_dir_path_passage=train_data_dir_path_passage,
    #                                                     dev_data_dir_path_passage=dev_data_dir_path_passage,
    #                                                     dev_qrels_path_passage=dev_qrels_path_passage,
    #                                                     dev_run_path_passage=dev_run_path_passage
    #
    #                                                     )
    #     else:
    #         experiment = FineTuningReRankingExperiments(model_path=model_path,
    #                                                     extra_layers=extra_layers,
    #                                                     train_batch_size=train_batch_size,
    #                                                     dev_batch_size=dev_batch_size,
    #
    #                                                     train_data_dir_path_entity=train_data_dir_path_entity,
    #                                                     dev_data_dir_path_entity=dev_data_dir_path_entity,
    #                                                     dev_qrels_path_entity=dev_qrels_path_entity,
    #                                                     dev_run_path_entity=dev_run_path_entity,
    #
    #                                                     )
    #
    #     rerank_run_path = '/nfs/trec_car/data/entity_ranking/testY1_hierarchical_passage_data/' + name
    #     experiment.inference(head_flag=head_flag, rerank_run_path=rerank_run_path, do_eval=False, cap_rank=1000)
    # # #
    # # from REL.mention_detection import MentionDetection
    # # from REL.utils import process_results
    # # from REL.entity_disambiguation import EntityDisambiguation
    # # from REL.ner import Cmns, load_flair_ner
    # # import time
    # #
    # # base_url = "/Users/iain/LocalStorage/coding/github/REL/"
    # # wiki_year = '2014'
    # # wiki_version = "wiki_" + wiki_year
    # # model_path = "/Users/iain/LocalStorage/coding/github/REL/ed-wiki-{}/model".format(wiki_year)
    #
    #
    # def example_preprocessing():
    #     # user does some stuff, which results in the format below.
    #     text_1 = "Roger Federer (German pronunciation: [ˈrɔdʒər ˈfeːdərər]; born 8 August 1981) is a Swiss professional tennis player who is ranked world No. 4 in men's singles tennis by the Association of Tennis Professionals (ATP).[3] He has won 20 Grand Slam singles titles—the most in history for a male player—and has held the world No. 1 spot in the ATP rankings for a record total of 310 weeks (including a record 237 consecutive weeks) and was the year-end No. 1 five times, including four consecutive. Federer, who turned professional in 1998, was continuously ranked in the top 10 from October 2002 to November 2016."
    #     text_2 = "Federer has won a record eight Wimbledon men's singles titles, six Australian Open titles, five US Open titles (all consecutive, a record), and one French Open title. He is one of eight men to have achieved a Career Grand Slam. Federer has reached a record 31 men's singles Grand Slam finals, including 10 consecutively from the 2005 Wimbledon Championships to the 2007 US Open. Federer has also won a record six ATP Finals titles, 28 ATP Tour Masters 1000 titles, and a record 24 ATP Tour 500 titles. Federer was a member of Switzerland's winning Davis Cup team in 2014. He is also the only player after Jimmy Connors to have won 100 or more career singles titles, as well as to amass 1,200 wins in the Open Era."
    #
    #     processed = {"test_doc1": [text_1, []],
    #                  "test_doc2": [text_2, []],
    #                  "test_doc3": [text_1, []],
    #                  "test_doc4": [text_2, []],
    #                  "test_doc5": [text_1, []],
    #                  "test_doc6": [text_2, []],
    #                  "test_doc7": [text_1, []],
    #                  "test_doc8": [text_2, []],
    #                  "test_doc9": [text_1, []],
    #                  "test_doc10": [text_2, []],
    #                  }
    #     return processed
    #
    #
    # start = time.time()
    # for i in range(1):
    #
    #
    #     input_text = example_preprocessing()
    #
    #     mention_detection = MentionDetection(base_url, wiki_version)
    #     tagger_ner = load_flair_ner("ner-fast")
    #     # tagger_ngram = Cmns(base_url, wiki_version, n=5)
    #     mentions_dataset, n_mentions = mention_detection.find_mentions(input_text, tagger_ner)
    #
    #     config = {
    #         "mode": "eval",
    #         "model_path": model_path,
    #     }
    #
    #     model = EntityDisambiguation(base_url, wiki_version, config)
    #     predictions, timing = model.predict(mentions_dataset)
    #
    #     result = process_results(mentions_dataset, predictions, input_text)
    #     print('*** result ***')
    #     print(result)

    # print(time.time() - start)
