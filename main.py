
import os

from learning.experiments import FineTuningReRankingExperiments
from retrieval.dataset_processing import TrecCarProcessing
from retrieval.tools import EvalTools

if __name__ == '__main__':
    # run_path =  '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmark_Y1_25_pyserini.run'
    # qrels_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmark_Y1_25.qrels'
    # eval_config = {
    #     'map': {'k': None},
    #     'Rprec': {'k': None},
    #     'recip_rank': {'k': None},
    #     'P': {'k': 20},
    #     'recall': {'k': 40},
    #     'ndcg': {'k': 20},
    # }
    #
    # eval = EvalTools()
    # eval.write_eval_from_qrels_and_run(run_path=run_path, qrels_path=qrels_path, eval_config=eval_config)

    # train_data_dir_path = '/nfs/trec_car/data/bert_reranker_datasets/train_benchmarkY1_chunks/'
    train_data_dir_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_train_entity_100_chucks/'
    train_batch_size = 8
    # dev_data_dir_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmark_Y1_25_chucks/'
    dev_data_dir_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_dev_entity_100_chucks/'
    dev_batch_size = 64
    dev_qrels_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_dev_entity.qrels'
    dev_run_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_dev_entity_100.run'
    # dev_qrels_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmark_Y1_25.qrels'
    # dev_run_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmark_Y1_25.run'
    model_path = None
    experiment = FineTuningReRankingExperiments(model_path=model_path,
                                                train_data_dir_path=train_data_dir_path,
                                                train_batch_size=train_batch_size,
                                                dev_data_dir_path=dev_data_dir_path,
                                                dev_batch_size=dev_batch_size,
                                                dev_qrels_path=dev_qrels_path,
                                                dev_run_path=dev_run_path)

    epochs = 3
    lr = 1e-5
    eps = 1e-8
    weight_decay = 0.01
    num_warmup_steps = 500
    experiments_dir = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    experiment_name = 'benchmarkY1_entity_100_retest'
    write = True
    logging_steps = 500
    head_flag = 'entity'

    experiment.run_experiment_single_head(
                                    head_flag=head_flag,
                                    epochs=epochs,
                                    lr=lr,
                                    eps=eps,
                                    weight_decay=weight_decay,
                                    num_warmup_steps=num_warmup_steps,
                                    experiments_dir=experiments_dir,
                                    experiment_name=experiment_name,
                                    logging_steps=logging_steps)
    # head_flag = 'passage'
    # rerank_run_path = '/nfs/trec_car/data/bert_reranker_datasets/test_runs/initial_new_pipeline_passage.run'
    # experiment.inference(head_flag=head_flag, rerank_run_path=rerank_run_path)