
import os

from metadata import EntityPaths, PassagePaths
from learning.experiments import FineTuningReRankingExperiments
from retrieval.dataset_processing import TrecCarProcessing
from retrieval.tools import EvalTools, SearchTools, default_eval_config

if __name__ == '__main__':

    # run_paths = ['/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_train_data/benchmarkY1_train_entity_synthetic_1000.run', '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity_synthetic_1000.run']
    # qrels_paths = ['/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_train_data/benchmarkY1_train_entity_synthetic.qrels', '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity_synthetic.qrels']
    # topics_paths = ['/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_train_data/benchmarkY1_train_entity_synthetic.topics', '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity_synthetic.topics']
    # data_dir_paths = ['/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_train_data/benchmarkY1_train_entity_synthetic_1000_chunks/', '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity_synthetic_1000_chunks/']
    # training_datasets = [True, False]
    # index_paths = [EntityPaths.index, EntityPaths.index]
    # hits = 1000
    # printing_step = 100
    # searcher_config = {
    #     'BM25': {'k1': 5.5, 'b': 0.1}
    # }
    # for run_path, qrels_path, topics_path, data_dir_path, training_dataset, index_path in zip(run_paths, qrels_paths, topics_paths, data_dir_paths, training_datasets, index_paths):
    #     print('searching')
    #     search = SearchTools(index_path=index_path, searcher_config=searcher_config)
    #     search.write_run_from_topics(topics_path=topics_path, run_path=run_path, hits=hits, printing_step=printing_step)
    #     print('eval')
    #     eval = EvalTools()
    #     eval.write_eval_from_qrels_and_run(run_path=run_path, qrels_path=qrels_path, eval_config=default_eval_config)
    #     print('dataset')
    #     processing = TrecCarProcessing(qrels_path=qrels_path,
    #                                    run_path=run_path,
    #                                    index_path=index_path,
    #                                    data_dir_path=data_dir_path)
    #
    #     processing.build_dataset(training_dataset=training_dataset, chuck_topic_size=50, first_para=True)
    train_data_dir_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_train_data/benchmarkY1_train_entity_synthetic_300_chunks/'
    train_batch_size = 12
    dev_data_dir_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity_synthetic_300_chunks/'
    dev_batch_size = 64 * 8
    dev_qrels_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity_synthetic.qrels'
    dev_run_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity_synthetic_300.run'
    model_path = None #'/nfs/trec_car/data/bert_reranker_datasets/exp/benchmarkY1_hierarchical_synthetic_passage_100_lr_8e6_num_warmup_steps_0.1/epoch1_batch3000/'
    experiment = FineTuningReRankingExperiments(model_path=model_path,
                                                train_data_dir_path=train_data_dir_path,
                                                train_batch_size=train_batch_size,
                                                dev_data_dir_path=dev_data_dir_path,
                                                dev_batch_size=dev_batch_size,
                                                dev_qrels_path=dev_qrels_path,
                                                dev_run_path=dev_run_path)

    epochs = 2
    lr = 5e-6
    eps = 1e-8
    weight_decay = 0.01
    warmup_percentage = 0.1
    experiments_dir = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    experiment_name = 'benchmarkY1_hierarchical_entity_300_lr_5e6_num_warmup_steps_0.1_new_pipeline'
    write = True
    logging_steps = 1000
    head_flag = 'entity'

    experiment.run_experiment_single_head(
                                    head_flag=head_flag,
                                    epochs=epochs,
                                    lr=lr,
                                    eps=eps,
                                    weight_decay=weight_decay,
                                    warmup_percentage=warmup_percentage,
                                    experiments_dir=experiments_dir,
                                    experiment_name=experiment_name,
                                    logging_steps=logging_steps)

    # head_flag = 'entity'
    # rerank_run_path = '/nfs/trec_car/data/entity_ranking/test_runs/benchmarkY1_hierarchical_entity_100_test_Y2_automatic_entity_v2.run'
    # experiment.inference(head_flag=head_flag, rerank_run_path=rerank_run_path)