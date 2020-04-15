
import os

from learning.experiments import FineTuningReRankingExperiments
from retrieval.dataset_processing import TrecCarProcessing

if __name__ == '__main__':

    index_path = '/nfs/trec_car/index/anserini_paragraphs/lucene-index.car17v2.0.paragraphsv2'
    run_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmark_Y1_25.run'
    qrels_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmark_Y1_25.qrels'
    data_dir_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmark_Y1_25_chucks/'


    processing = TrecCarProcessing(qrels_path=qrels_path,
                                   run_path=run_path,
                                   index_path=index_path,
                                   data_dir_path=data_dir_path)

    processing.build_dataset(sequential=True, chuck_topic_size=250)



    # train_data_dir_path = '/nfs/trec_car/data/bert_reranker_datasets/test_chunks_train/'
    # train_batch_size = 8
    # dev_data_dir_path = '/nfs/trec_car/data/bert_reranker_datasets/test_chunks/'
    # dev_batch_size = 64
    # dev_qrels_path = '/nfs/trec_car/data/bert_reranker_datasets/test_10.qrels'
    # dev_run_path = '/nfs/trec_car/data/bert_reranker_datasets/test_10.run'
    # experiment = FineTuningReRankingExperiments(train_data_dir_path=train_data_dir_path,
    #                                             train_batch_size=train_batch_size,
    #                                             dev_data_dir_path=dev_data_dir_path,
    #                                             dev_batch_size=dev_batch_size,
    #                                             dev_qrels_path=dev_qrels_path,
    #                                             dev_run_path=dev_run_path)
    #
    # epochs = 3
    # lr = 2e-5
    # eps = 1e-8
    # weight_decay = 0.01
    # num_warmup_steps = 0
    # experiments_dir = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    # experiment_name = 'testing_training_and_validation'
    # write = True
    # logging_steps = 100
    #
    # experiment.run_experiment_single_head(
    #                                 head_flag='entity',
    #                                 epochs=epochs,
    #                                 lr=lr,
    #                                 eps=eps,
    #                                 weight_decay=weight_decay,
    #                                 num_warmup_steps=num_warmup_steps,
    #                                 experiments_dir=experiments_dir,
    #                                 experiment_name=experiment_name,
    #                                 logging_steps=logging_steps)
