
import os

from metadata import EntityPaths, PassagePaths
from learning.experiments import FineTuningReRankingExperiments
from learning.models import RoBERTaMultiTaskRanker
from retrieval.dataset_processing import TrecCarProcessing, RobertaTokenizer
from retrieval.tools import EvalTools, SearchTools, default_eval_config
from torch import nn

if __name__ == '__main__':

    # index_path = PassagePaths.index
    # printing_step = 100
    # searcher_config = {
    #     'BM25': {'k1': 0.9, 'b': 0.4}
    # }
    # max_length = 512
    # use_token_type_ids = False
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    #
    # run_path = '/nfs/trec_car/data/passage_ranking/testY1_hierarchical_passage_1000.run'
    # qrels_path = '/nfs/trec_car/data/passage_ranking/testY1_hierarchical_passage.qrels'
    # topics_path = '/nfs/trec_car/data/passage_ranking/testY1_hierarchical_passage.topics'
    # data_dir_path = '/nfs/trec_car/data/passage_ranking/testY1_hierarchical_passage_1000_roberta_chunks/'
    # training_dataset = False
    # hits = 1000

    #
    # search = SearchTools(index_path=index_path, searcher_config=searcher_config)
    # # print('building topics')
    # # search.write_topics_from_qrels(qrels_path=qrels_path, topics_path=topics_path)
    # print('** searching **')
    # search.write_run_from_topics(topics_path=topics_path, run_path=run_path, hits=hits, printing_step=printing_step)
    # print('** eval **')
    # eval = EvalTools()
    # eval.write_eval_from_qrels_and_run(run_path=run_path, qrels_path=qrels_path, eval_config=default_eval_config)
    # print('** dataset **')
    # processing = TrecCarProcessing(qrels_path=qrels_path,
    #                                run_path=run_path,
    #                                index_path=index_path,
    #                                data_dir_path=data_dir_path,
    #                                use_token_type_ids=use_token_type_ids,
    #                                tokenizer=tokenizer,
    #                                max_length=max_length)
    #
    # processing.build_dataset(training_dataset=training_dataset, chuck_topic_size=50, first_para=False)


    train_data_dir_path = '/nfs/trec_car/data/passage_ranking/dtrain_benchmarkY1_250_roberta_chunks/'
    train_batch_size = 12
    dev_batch_size = 64 * 8
    dev_data_dir_path = '/nfs/trec_car/data/passage_ranking/dev_benchmark_Y1_25_roberta_chunks/'
    dev_qrels_path = '/nfs/trec_car/data/passage_ranking/dev_benchmark_Y1_25.qrels'
    dev_run_path = '/nfs/trec_car/data/passage_ranking/dev_benchmark_Y1_25.run'
    model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/roberta_benchmarkY1_lr_6e6_v2/epoch1_batch3000' #None
    use_token_type_ids = False
    experiment = FineTuningReRankingExperiments(model_path=model_path,
                                                use_token_type_ids=use_token_type_ids,
                                                train_data_dir_path=train_data_dir_path,
                                                train_batch_size=train_batch_size,
                                                dev_data_dir_path=dev_data_dir_path,
                                                dev_batch_size=dev_batch_size,
                                                dev_qrels_path=dev_qrels_path,
                                                dev_run_path=dev_run_path)

    epochs = 2
    lr = 2e-5
    eps = 1e-8
    weight_decay = 0.01
    warmup_percentage = 0.1
    experiments_dir = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    experiment_name = 'roberta_benchmarkY1_lr_2e5_v3'
    write = True
    logging_steps = 500
    head_flag = 'passage'

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

    # head_flag = 'passage'
    # rerank_run_path = '/nfs/trec_car/data/entity_ranking/test_runs/roberta_dev_test_write.run'
    # experiment.inference(head_flag=head_flag, rerank_run_path=rerank_run_path, do_eval=True)