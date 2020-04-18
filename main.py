
import os

from metadata import EntityPaths, PassagePaths
from learning.experiments import FineTuningReRankingExperiments
from retrieval.dataset_processing import TrecCarProcessing
from retrieval.tools import EvalTools, SearchTools

if __name__ == '__main__':
    index_path = PassagePaths.index
    run_paths = ['/nfs/trec_car/data/entity_ranking/benchmarkY1_train_passage_100.run', '/nfs/trec_car/data/entity_ranking/benchmarkY1_dev_passage_100.run']
    qrels_paths = ['/nfs/trec_car/data/entity_ranking/benchmarkY1_train_passage.qrels', '/nfs/trec_car/data/entity_ranking/benchmarkY1_dev_passage.qrels']
    topics_paths = ['/nfs/trec_car/data/entity_ranking/benchmarkY1_train_passage.topics', '/nfs/trec_car/data/entity_ranking/benchmarkY1_dev_passage.topics']
    data_dir_paths = ['/nfs/trec_car/data/entity_ranking/benchmarkY1_train_passage_100_chunks/', '/nfs/trec_car/data/entity_ranking/benchmarkY1_dev_passage_100_chunks/']
    training_datasets = [True, False]
    hits = 100
    printing_step = 100


    searcher_config = {
        'BM25': {'k1': 0.9, 'b': 0.4}
    }
    eval_config = {
        'map': {'k': None},
        'Rprec': {'k': None},
        'recip_rank': {'k': None},
        'P': {'k': 20},
        'recall': {'k': 40},
        'ndcg': {'k': 20},
    }
    for run_path, qrels_path, topics_path, data_dir_path, training_dataset in zip(run_paths, qrels_paths, topics_paths, data_dir_paths, training_datasets):

        search = SearchTools(index_path=index_path, searcher_config=searcher_config)
        search.write_run_from_topics(topics_path=topics_path, run_path=run_path, hits=hits, printing_step=printing_step)

        eval = EvalTools()
        eval.write_eval_from_qrels_and_run(run_path=run_path, qrels_path=qrels_path, eval_config=eval_config)

        processing = TrecCarProcessing(qrels_path=qrels_path,
                                       run_path=run_path,
                                       index_path=index_path,
                                       data_dir_path=data_dir_path)

        processing.build_dataset(training_dataset=training_dataset, chuck_topic_size=100)


    # # train_data_dir_path = '/nfs/trec_car/data/bert_reranker_datasets/train_benchmarkY1_chunks/'
    # train_data_dir_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_train_entity_100_chucks/'
    # train_batch_size = 8
    # # dev_data_dir_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmark_Y1_25_chucks/'
    # dev_data_dir_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_dev_entity_100_chucks/'
    # dev_batch_size = 64
    # dev_qrels_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_dev_entity.qrels'
    # dev_run_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_dev_entity_100.run'
    # # dev_qrels_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmark_Y1_25.qrels'
    # # dev_run_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmark_Y1_25.run'
    # model_path = None
    # experiment = FineTuningReRankingExperiments(model_path=model_path,
    #                                             train_data_dir_path=train_data_dir_path,
    #                                             train_batch_size=train_batch_size,
    #                                             dev_data_dir_path=dev_data_dir_path,
    #                                             dev_batch_size=dev_batch_size,
    #                                             dev_qrels_path=dev_qrels_path,
    #                                             dev_run_path=dev_run_path)
    #
    # epochs = 3
    # lr = 8e-6
    # eps = 1e-8
    # weight_decay = 0.01
    # num_warmup_steps = 1000
    # experiments_dir = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    # experiment_name = 'benchmarkY1_entity_100_lr_8e6_num_warmup_steps_1000'
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
    # head_flag = 'passage'
    # rerank_run_path = '/nfs/trec_car/data/bert_reranker_datasets/test_runs/initial_new_pipeline_passage.run'
    # experiment.inference(head_flag=head_flag, rerank_run_path=rerank_run_path)