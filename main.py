
import os

from metadata import EntityPaths, PassagePaths
from learning.experiments import FineTuningReRankingExperiments
from retrieval.dataset_processing import TrecCarProcessing
from retrieval.tools import EvalTools, SearchTools

if __name__ == '__main__':

    # run_paths = ['/nfs/trec_car/data/entity_ranking/benchmarkY1_tree_passage_train_data/benchmarkY1_tree_passage_train_100.run', '/nfs/trec_car/data/entity_ranking/benchmarkY1_tree_passage_dev_data/benchmarkY1_tree_passage_dev_100.run', '/nfs/trec_car/data/entity_ranking/benchmarkY1_tree_no_root_passage_train_data/benchmarkY1_tree_no_root_passage_train_100.run', '/nfs/trec_car/data/entity_ranking/benchmarkY1_tree_no_root_passage_dev_data/benchmarkY1_tree_no_root_passage_dev_100.run']
    # qrels_paths = ['/nfs/trec_car/data/entity_ranking/benchmarkY1_tree_passage_train_data/benchmarkY1_tree_passage_train.qrels', '/nfs/trec_car/data/entity_ranking/benchmarkY1_tree_passage_dev_data/benchmarkY1_tree_passage_dev.qrels', '/nfs/trec_car/data/entity_ranking/benchmarkY1_tree_no_root_passage_train_data/benchmarkY1_tree_no_root_passage_train.qrels', '/nfs/trec_car/data/entity_ranking/benchmarkY1_tree_no_root_passage_dev_data/benchmarkY1_tree_no_root_passage_dev.qrels']
    # topics_paths = ['/nfs/trec_car/data/entity_ranking/benchmarkY1_tree_passage_train_data/benchmarkY1_tree_passage_train.topics', '/nfs/trec_car/data/entity_ranking/benchmarkY1_tree_passage_dev_data/benchmarkY1_tree_passage_dev.topics', '/nfs/trec_car/data/entity_ranking/benchmarkY1_tree_no_root_passage_train_data/benchmarkY1_tree_no_root_passage_train.topics', '/nfs/trec_car/data/entity_ranking/benchmarkY1_tree_no_root_passage_dev_data/benchmarkY1_tree_no_root_passage_dev.topics']
    # data_dir_paths = ['/nfs/trec_car/data/entity_ranking/benchmarkY1_tree_passage_train_data/benchmarkY1_tree_passage_train_100_chunks/','/nfs/trec_car/data/entity_ranking/benchmarkY1_tree_passage_dev_data/benchmarkY1_tree_passage_dev_100_chunks/', '/nfs/trec_car/data/entity_ranking/benchmarkY1_tree_no_root_passage_train_data/benchmarkY1_tree_no_root_passage_train_100_chunks/','/nfs/trec_car/data/entity_ranking/benchmarkY1_tree_no_root_passage_dev_data/benchmarkY1_tree_no_root_passage_dev_100_chunks/']
    # training_datasets = [True, False, True, False]

    # run_paths = ['/nfs/trec_car/data/entity_ranking/testY2_automatic_entity_data/testY2_automatic_entity_1000.run', '/nfs/trec_car/data/entity_ranking/testY2_manual_entity_data/testY2_manual_entity_1000.run', '/nfs/trec_car/data/entity_ranking/testY2_manual_passage_data/testY2_manual_passage_1000.run']
    # qrels_paths = ['/nfs/trec_car/data/entity_ranking/testY2_automatic_entity_data/testY2_automatic_entity.qrels', '/nfs/trec_car/data/entity_ranking/testY2_manual_entity_data/testY2_manual_entity.qrels', '/nfs/trec_car/data/entity_ranking/testY2_manual_passage_data/testY2_manual_passage.qrels']
    # topics_paths = ['/nfs/trec_car/data/entity_ranking/testY2_automatic_entity_data/testY2_automatic_entity.topics', '/nfs/trec_car/data/entity_ranking/testY2_manual_entity_data/testY2_manual_entity.topics', '/nfs/trec_car/data/entity_ranking/testY2_manual_passage_data/testY2_manual_passage.topics']
    # data_dir_paths = ['/nfs/trec_car/data/entity_ranking/testY2_automatic_entity_data/testY2_automatic_entity_1000_chunks', '/nfs/trec_car/data/entity_ranking/testY2_manual_entity_data/testY2_manual_entity_1000_chunks', '/nfs/trec_car/data/entity_ranking/testY2_manual_passage_data/testY2_manual_passage_1000_chunks']
    # training_datasets = [False, False, False]
    # index_paths = [EntityPaths.index, EntityPaths.index, PassagePaths.index]

    # hits = 100
    # printing_step = 100
    # searcher_config = {
    #     'BM25': {'k1': 0.9, 'b': 0.4}
    # }
    # eval_config = {
    #     'map': {'k': None},
    #     'Rprec': {'k': None},
    #     'recip_rank': {'k': None},
    #     'P': {'k': 20},
    #     'recall': {'k': 40},
    #     'ndcg': {'k': 20},
    # }
    #
    # for run_path, qrels_path, topics_path, data_dir_path, training_dataset, index_path in zip(run_paths, qrels_paths, topics_paths, data_dir_paths, training_datasets, index_paths):
        # print('searching')
        # search = SearchTools(index_path=index_path, searcher_config=searcher_config)
        # search.write_run_from_topics(topics_path=topics_path, run_path=run_path, hits=hits, printing_step=printing_step)
        # print('eval')
        # eval = EvalTools()
        # eval.write_eval_from_qrels_and_run(run_path=run_path, qrels_path=qrels_path, eval_config=eval_config)
        # print('dataset')
        # processing = TrecCarProcessing(qrels_path=qrels_path,
        #                                run_path=run_path,
        #                                index_path=index_path,
        #                                data_dir_path=data_dir_path)
        #
        # processing.build_dataset(training_dataset=training_dataset, chuck_topic_size=50, first_para=True)
    #
    train_data_dir_path = None #'/nfs/trec_car/data/entity_ranking/benchmarkY1_tree_no_root_passage_train_data/benchmarkY1_tree_no_root_passage_train_100_chunks/'
    train_batch_size = None #8
    dev_data_dir_path = '/nfs/trec_car/data/entity_ranking/testY2_manual_passage_data/testY2_manual_passage_1000_chunks/'
    dev_batch_size = 64 * 8
    dev_qrels_path = '/nfs/trec_car/data/entity_ranking/testY2_manual_passage_data/testY2_manual_passage.qrels'
    dev_run_path = '/nfs/trec_car/data/entity_ranking/testY2_manual_passage_data/testY2_manual_passage_1000.run'
    model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/benchmarkY1_passage_100_lr_8e6_num_warmup_steps_1000/epoch1_batch14000/'
    experiment = FineTuningReRankingExperiments(model_path=model_path,
                                                train_data_dir_path=train_data_dir_path,
                                                train_batch_size=train_batch_size,
                                                dev_data_dir_path=dev_data_dir_path,
                                                dev_batch_size=dev_batch_size,
                                                dev_qrels_path=dev_qrels_path,
                                                dev_run_path=dev_run_path)

    # epochs = 2
    # lr = 8e-6
    # eps = 1e-8
    # weight_decay = 0.01
    # warmup_percentage = 0.1
    # experiments_dir = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    # experiment_name = 'benchmarkY1_tree_no_root_passage_100_lr_8e6_num_warmup_steps_0.1'
    # write = True
    # logging_steps = 1000
    # head_flag = 'passage'
    #
    # experiment.run_experiment_single_head(
    #                                 head_flag=head_flag,
    #                                 epochs=epochs,
    #                                 lr=lr,
    #                                 eps=eps,
    #                                 weight_decay=weight_decay,
    #                                 warmup_percentage=warmup_percentage,
    #                                 experiments_dir=experiments_dir,
    #                                 experiment_name=experiment_name,
    #                                 logging_steps=logging_steps)

    head_flag = 'passage'
    rerank_run_path = '/nfs/trec_car/data/entity_ranking/test_runs/benchmarkY1_hierarchical_passage_100_model_test_Y2_manual_passage.run'
    experiment.inference(head_flag=head_flag, rerank_run_path=rerank_run_path)