
import os

from metadata import EntityPaths, PassagePaths
from learning.experiments import FineTuningReRankingExperiments
from retrieval.dataset_processing import TrecCarProcessing, BertTokenizer
from retrieval.tools import EvalTools, SearchTools, default_eval_config
from torch import nn

if __name__ == '__main__':

    index_path = '/home/iain_mackie1993/nfs/data/paragraphs_corpus/index/anserini.paragraph.index.v5'
    printing_step = 1000
    searcher_config = {
        'BM25': {'k1': 0.9, 'b': 0.4}
    }
    max_length = 512

    run_path = '/home/iain_mackie1993/nfs/data/trec_car/passage_data/fold-0-base.train.cbor-hierarchical_250.run'
    qrels_path = '/nfs/trec_car/data/entity_ranking/train_passage_fold0/fold-0-base.train.cbor-hierarchical.qrels'
    topics_path = '/home/iain_mackie1993/nfs/data/trec_car/passage_data/fold-0-base.train.cbor-hierarchical.topics'
    data_dir_path = '/home/iain_mackie1993/nfs/data/trec_car/passage_data/benchmarkY1_passage_bert_250_chunks/'
    training_dataset = True
    hits = 250

    search = SearchTools(index_path=index_path, searcher_config=searcher_config)
    print('building topics')
    search.write_topics_from_qrels(qrels_path=qrels_path, topics_path=topics_path)
    print('** searching **')
    search.write_run_from_topics(topics_path=topics_path, run_path=run_path, hits=hits, printing_step=printing_step)
    print('** eval **')
    eval = EvalTools()
    eval.write_eval_from_qrels_and_run(run_path=run_path, qrels_path=qrels_path, eval_config=default_eval_config)
    print('** dataset **')
    processing = TrecCarProcessing(qrels_path=qrels_path,
                                   run_path=run_path,
                                   index_path=index_path,
                                   data_dir_path=data_dir_path,
                                   max_length=max_length)

    processing.build_dataset(training_dataset=training_dataset, chuck_topic_size=500, first_para=False)

    # gpus = 2
    # train_data_dir_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_passage_train_data/benchmarkY1_hierarchical_passage_train_100_chunks/'
    # train_batch_size = 8 * gpus
    # dev_batch_size = 64 * 2 * gpus
    # dev_data_dir_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_passage_dev_data/benchmarkY1_hierarchical_passage_dev_100_chunks/'
    # dev_qrels_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_passage_dev_data/benchmarkY1_dev_passage.qrels'
    # dev_run_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_passage_dev_data/benchmarkY1_dev_passage_100.run'
    # model_path = None #'/nfs/trec_car/data/bert_reranker_datasets/exp/roberta_benchmarkY1_lr_6e6_v2/epoch1_batch3000' #None
    # experiment = FineTuningReRankingExperiments(model_path=model_path,
    #                                             train_data_dir_path=train_data_dir_path,
    #                                             train_batch_size=train_batch_size,
    #                                             dev_data_dir_path=dev_data_dir_path,
    #                                             dev_batch_size=dev_batch_size,
    #                                             dev_qrels_path=dev_qrels_path,
    #                                             dev_run_path=dev_run_path)
    #
    # epochs = 2
    # lr = 8e-6
    # eps = 1e-8
    # weight_decay = 0.01
    # warmup_percentage = 0.1
    # experiments_dir = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    # experiment_name = 'bert_test_pipeline_working_remove_roberta'
    # write = True
    # logging_steps = 250
    # head_flag = 'passage'
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

    # head_flag = 'passage'
    # rerank_run_path = '/nfs/trec_car/data/entity_ranking/test_runs/roberta_dev_test_write.run'
    # experiment.inference(head_flag=head_flag, rerank_run_path=rerank_run_path, do_eval=True)