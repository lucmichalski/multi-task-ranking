
import os

from metadata import EntityPaths, PassagePaths
from learning.experiments import FineTuningReRankingExperiments
from retrieval.dataset_processing import TrecCarProcessing, BertTokenizer
from retrieval.tools import EvalTools, SearchTools, default_eval_config
from torch import nn

if __name__ == '__main__':

    index_path = EntityPaths.index #'/home/iain_mackie1993/nfs/data/paragraphs_corpus/index/anserini.paragraph.index.v5'
    printing_step = 500
    searcher_config = {
        'BM25': {'k1': 5.5, 'b': 0.1}
    }
    max_length = 512
    run_paths = ['/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_train_data/benchmarkY1_train_entity_synthetic_300.run', '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity_synthetic_100.run']
    qrels_paths = ['/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_train_data/benchmarkY1_train_entity_synthetic.qrels', '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity_synthetic.qrels']
    data_dir_paths = ['/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_train_data/benchmarkY1_train_entity_synthetic_300_chunks_context_v2/', '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity_synthetic_100_chunks_context_v2/']
    training_datasets = [True, False]
    context_paths = ['/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_train_data/benchmarkY1_train_entity_synthetic_300.run.context_with_special_tokens.json', '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity_synthetic_100.run.context_with_special_tokens.json']
    hits = [300, 25]
    #
    # search = SearchTools(index_path=index_path, searcher_config=searcher_config)
    # print('building topics')
    # search.write_topics_from_qrels(qrels_path=qrels_path, topics_path=topics_path)
    # print('** searching **')
    # search.write_run_from_topics(topics_path=topics_path, run_path=run_path, hits=hits, printing_step=printing_step)
    # print('** eval **')
    # eval = EvalTools()
    # eval.write_eval_from_qrels_and_run(run_path=run_path, qrels_path=qrels_path, eval_config=default_eval_config)
    # print('** dataset **')
    # for run_path, qrels_path, data_dir_path, training_dataset, context_path in zip(run_paths, qrels_paths, data_dir_paths, training_datasets, context_paths):
    #     processing = TrecCarProcessing(qrels_path=qrels_path,
    #                                    run_path=run_path,
    #                                    index_path=index_path,
    #                                    data_dir_path=data_dir_path,
    #                                    max_length=max_length,
    #                                    context_path=context_path)
    #
    #     processing.build_dataset(training_dataset=training_dataset, chuck_topic_size=250, first_para=False)

    gpus = 5
    train_data_dir_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_train_data/benchmarkY1_train_entity_synthetic_300_chunks_context_v2/'
    train_batch_size = 8 * gpus
    dev_batch_size = 64 * 2 * gpus
    dev_data_dir_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity_synthetic_100_chunks_context_v2/'
    dev_qrels_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity_synthetic.qrels'
    dev_run_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_entity_dev_data/benchmarkY1_dev_entity_synthetic_100.run'
    model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/roberta_benchmarkY1_lr_6e6_v2/epoch1_batch3000' #None
    experiment = FineTuningReRankingExperiments(model_path=model_path,
                                                train_data_dir_path=train_data_dir_path,
                                                train_batch_size=train_batch_size,
                                                dev_data_dir_path=dev_data_dir_path,
                                                dev_batch_size=dev_batch_size,
                                                dev_qrels_path=dev_qrels_path,
                                                dev_run_path=dev_run_path)

    epochs = 2
    lr = 1e-5
    eps = 1e-8
    weight_decay = 0.01
    warmup_percentage = 0.1
    experiments_dir = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    experiment_name = 'bert_with_context_no_ent_v1_8e6'
    write = True
    logging_steps = 500
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
        logging_steps=logging_steps
    )

    # head_flag = 'entity'
    # rerank_run_path = '/nfs/trec_car/data/entity_ranking/test_runs/roberta_dev_test_write.run'
    # experiment.inference(head_flag=head_flag, rerank_run_path=rerank_run_path, do_eval=True)