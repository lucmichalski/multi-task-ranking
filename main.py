
import os

from metadata import EntityPaths, PassagePaths
from learning.experiments import FineTuningReRankingExperiments
from retrieval.dataset_processing import TrecCarProcessing
from retrieval.tools import EvalTools, SearchTools, default_eval_config

if __name__ == '__main__':

    index_path = PassagePaths.index
    hits = 1000
    printing_step = 100
    searcher_config = {
        'BM25': {'k1': 0.9, 'b': 0.4}
    }

    run_path = '/nfs/trec_car/data/entity_ranking/using_passage/test_entity_Y1_passage.run'
    qrels_path = '/nfs/trec_car/data/entity_ranking/testY2_manual_entity_data/testY2_manual_entity.qrels'
    topics_path = '/nfs/trec_car/data/entity_ranking/testY2_manual_entity_data/testY2_manual_entity.topics'
    data_dir_path = '/nfs/trec_car/data/entity_ranking/using_passage/test_entity_Y1_passage_1000_hits_50_chunks/'
    training_dataset = False

    search = SearchTools(index_path=index_path, searcher_config=searcher_config)
    # print('building topics')
    # search.write_topics_from_qrels(qrels_path=qrels_path, topics_path=topics_path)
    # print('searching')
    # search.write_run_from_topics(topics_path=topics_path, run_path=run_path, hits=hits, printing_step=printing_step)
    # print('eval')
    # eval = EvalTools()
    # eval.write_eval_from_qrels_and_run(run_path=run_path, qrels_path=qrels_path, eval_config=default_eval_config)
    print('dataset')
    processing = TrecCarProcessing(qrels_path=qrels_path,
                                   run_path=run_path,
                                   index_path=index_path,
                                   data_dir_path=data_dir_path)
#
    processing.build_dataset(training_dataset=training_dataset, chuck_topic_size=50, first_para=False)

#     for doc_count in [10, 100, 1000, 10000]:
#         print('-------------------------------')
#         print('RUNNING FOR DOC COUNT {}'.format(doc_count))
#
#         index_path = EntityPaths.index
#         hits = 300
#         printing_step = 100
#         searcher_config = {
#             'BM25': {'k1': 5.5, 'b': 0.1}
#         }
#
#         run_path = '/nfs/trec_car/data/entity_ranking/full_data/full_data_v2_hierarchical_{}_hits_{}.run'.format(doc_count, hits)
#         qrels_path = '/nfs/trec_car/data/entity_ranking/full_data/full_data_v2_hierarchical_{}.qrels'.format(doc_count)
#         topics_path = '/nfs/trec_car/data/entity_ranking/full_data/full_data_v2_hierarchical_{}.topics'.format(doc_count)
#         data_dir_path = '/nfs/trec_car/data/entity_ranking/full_data/full_data_v2_hierarchical_{}_hits_{}_chunks/'.format(doc_count, hits)
#         training_dataset = True
#
#         search = SearchTools(index_path=index_path, searcher_config=searcher_config)
#         print('building topics')
#         search.write_topics_from_qrels(qrels_path=qrels_path, topics_path=topics_path)
#         print('searching')
#         search.write_run_from_topics(topics_path=topics_path, run_path=run_path, hits=hits, printing_step=printing_step)
#         print('eval')
#         eval = EvalTools()
#         eval.write_eval_from_qrels_and_run(run_path=run_path, qrels_path=qrels_path, eval_config=default_eval_config)
#         print('dataset')
#         processing = TrecCarProcessing(qrels_path=qrels_path,
#                                        run_path=run_path,
#                                        index_path=index_path,
#                                        data_dir_path=data_dir_path)
# #
#         processing.build_dataset(training_dataset=training_dataset, chuck_topic_size=100, first_para=True)

    train_data_dir_path = None
    train_batch_size = None
    dev_batch_size = 64 * 8
    dev_data_dir_path = data_dir_path
    dev_qrels_path = qrels_path
    dev_run_path = run_path
    model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/benchmarkY1_passage_100_lr_8e6_num_warmup_steps_1000/epoch1_batch14000'
    experiment = FineTuningReRankingExperiments(model_path=model_path,
                                                train_data_dir_path=train_data_dir_path,
                                                train_batch_size=train_batch_size,
                                                dev_data_dir_path=dev_data_dir_path,
                                                dev_batch_size=dev_batch_size,
                                                dev_qrels_path=dev_qrels_path,
                                                dev_run_path=dev_run_path)


    # epochs = 2
    # lr = 6e-6
    # eps = 1e-8
    # weight_decay = 0.01
    # warmup_percentage = 0.1
    # experiments_dir = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    # experiment_name = 'full_data_v2_hierarchical_1000_hits_300_v2_lr_6e6_num_warmup_steps_0.1_new_pipeline'
    # write = True
    # logging_steps = 5000
    # head_flag = 'entity'
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
    rerank_run_path = '/nfs/trec_car/data/entity_ranking/test_runs/testY2_manual_entity_passages_to_entity.run'
    experiment.inference(head_flag=head_flag, rerank_run_path=rerank_run_path)