
import os

from metadata import EntityPaths, PassagePaths
from learning.experiments import FineTuningReRankingExperiments
from retrieval.dataset_processing import TrecCarProcessing
from retrieval.tools import EvalTools, SearchTools

if __name__ == '__main__':

    # run_paths = ['/nfs/trec_car/data/entity_ranking/test_hierarchical_entity_1000.run']
    # qrels_paths = ['/nfs/trec_car/data/entity_ranking/test_hierarchical_entity.qrels']
    # data_dir_paths = ['/nfs/trec_car/data/entity_ranking/test_hierarchical_entity_para_1000_chunks_v2/']
    # training_datasets = [False]
    #
    # index_path = EntityPaths.index
    # hits = 1000
    # printing_step = 100
    # searcher_config = {
    #     'BM25': {'k1': 5.5, 'b': 0.1}
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
    # for run_path, qrels_path, data_dir_path, training_dataset in zip(run_paths, qrels_paths, data_dir_paths, training_datasets):
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

    # train_data_dir_path = None #'/nfs/trec_car/data/entity_ranking/benchmarkY1_train_entity_para_100_chunks/'
    # train_batch_size = None #8
    # dev_data_dir_path = '/nfs/trec_car/data/entity_ranking/test_hierarchical_entity_para_1000_chunks/'
    # dev_batch_size = 64 * 8
    # dev_qrels_path = EntityPaths.test_qrels
    # dev_run_path = '/nfs/trec_car/data/entity_ranking/test_hierarchical_entity_1000.run'
    # model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/benchmarkY1_entity_100_lr_8e6_num_warmup_steps_2000_just_para/epoch1_batch14000/'
    # experiment = FineTuningReRankingExperiments(model_path=model_path,
    #                                             train_data_dir_path=train_data_dir_path,
    #                                             train_batch_size=train_batch_size,
    #                                             dev_data_dir_path=dev_data_dir_path,
    #                                             dev_batch_size=dev_batch_size,
    #                                             dev_qrels_path=dev_qrels_path,
    #                                             dev_run_path=dev_run_path)

    train_data_dir_path = None #'/nfs/trec_car/data/entity_ranking/benchmarkY1_train_entity_para_100_chunks/'
    train_batch_size = None #8
    dev_data_dir_path = '/nfs/trec_car/data/entity_ranking/test_hierarchical_passage_1000_chunks/'
    dev_batch_size = 64 * 8
    dev_qrels_path = PassagePaths.test_qrels
    dev_run_path = '/nfs/trec_car/data/entity_ranking/test_hierarchical_passage_1000.run'
    model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/benchmarkY1_passage_100_lr_8e6_num_warmup_steps_1000/epoch1_batch11000/'
    experiment = FineTuningReRankingExperiments(model_path=model_path,
                                                train_data_dir_path=train_data_dir_path,
                                                train_batch_size=train_batch_size,
                                                dev_data_dir_path=dev_data_dir_path,
                                                dev_batch_size=dev_batch_size,
                                                dev_qrels_path=dev_qrels_path,
                                                dev_run_path=dev_run_path)
    # epochs = 3
    # lr = 8e-6
    # eps = 1e-8
    # weight_decay = 0.01
    # num_warmup_steps = 2000
    # experiments_dir = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    # experiment_name = 'benchmarkY1_entity_100_lr_8e6_num_warmup_steps_2000_just_para'
    # write = True
    # logging_steps = 1000
    # head_flag = 'entity'
    #
    # experiment.run_experiment_single_head(
    #                                 head_flag=head_flag,
    #                                 epochs=epochs,
    #                                 lr=lr,
    #                                 eps=eps,
    #                                 weight_decay=weight_decay,
    #                                 num_warmup_steps=num_warmup_steps,
    #                                 experiments_dir=experiments_dir,
    #                                 experiment_name=experiment_name,
    #                                 logging_steps=logging_steps)
    # head_flag = 'entity'
    # rerank_run_path = '/nfs/trec_car/data/entity_ranking/test_runs/benchmarkY1_entity_100_lr_8e6_num_warmup_steps_2000_just_para_test_v2.run'
    # experiment.inference(head_flag=head_flag, rerank_run_path=rerank_run_path)

    head_flag = 'passage'
    rerank_run_path = '/nfs/trec_car/data/entity_ranking/test_runs/benchmarkY1_passage_100_lr_8e6_num_warmup_steps_1000_test_v2.run'
    experiment.inference(head_flag=head_flag, rerank_run_path=rerank_run_path)