
from multi_task.processing import MultiTaskDatasetByQuery
from multi_task.ranking import  train_cls_model_max_combo, train_cls_model, train_mutant_max_combo, train_mutant_multi_task_max_combo

if __name__ == '__main__':
    # dir_path = '/nfs/trec_car/data/entity_ranking/multi_task_data_by_query/'
    # passage_model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/benchmarkY1_passage_100_lr_8e6_num_warmup_steps_1000/epoch1_batch14000/'
    # entity_model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/full_data_v2_hierarchical_10000_hits_300_v2_lr_2e6_num_warmup_steps_0.1_new_pipeline/epoch1_batch420000/'
    # batch_size = 64*6
    # max_rank = 100
    # MultiTaskDatasetByQuery().build_dataset_by_query_entity_context(dir_path=dir_path,
    #                                                                 max_rank=max_rank,
    #                                                                 batch_size=batch_size,
    #                                                                 passage_model_path=passage_model_path,
    #                                                                 entity_model_path=entity_model_path
    #                                                                 )

    # train_mutant_max_combo()
    batch_sizes = [64, 256]
    lrs = [0.00001, 0.0001, 0.0005, 0.001]
    for batch_size in batch_sizes:
        for lr in lrs:
            train_cls_model(batch_size=batch_size, lr=lr)
            train_mutant_multi_task_max_combo(batch_size=batch_size, lr=lr)


