
from multi_task.processing import MultiTaskDatasetByQuery
from multi_task.ranking import rerank_runs, train_cls_model_max_combo, train_cls_model

if __name__ == '__main__':
    dir_path = '/nfs/trec_car/data/entity_ranking/multi_task_data_by_query/'
    # passage_model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/benchmarkY1_passage_100_lr_8e6_num_warmup_steps_1000/epoch1_batch14000/'
    # entity_model_path = '/nfs/trec_car/data/bert_reranker_datasets/exp/full_data_v2_hierarchical_10000_hits_300_v2_lr_2e6_num_warmup_steps_0.1_new_pipeline/epoch1_batch420000/'
    # MultiTaskDatasetByQuery().build_dataset_by_query(dir_path=dir_path,
    #                                                  max_rank=100,
    #                                                  batch_size=64,
    #                                                  bi_encode=True,
    #                                                  passage_model_path=passage_model_path,
    #                                                  entity_model_path=entity_model_path
    #                                                  )

    # dataset = 'test'
    # rerank_runs(dataset=dataset)

    train_cls_model()
    # train_cls_model(bi_encode=True)
    # train_cls_model_max_combo()

