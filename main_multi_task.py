
from multi_task.processing import MultiTaskDatasetByQuery
from multi_task.ranking import rerank_runs, train_cls_model_max_combo

if __name__ == '__main__':
    # dir_path = '/nfs/trec_car/data/entity_ranking/multi_task_data_by_query/'
    # MultiTaskDatasetByQuery().build_dataset_by_query(dir_path=dir_path)
    #
    # dataset = 'test'
    # rerank_runs(dataset=dataset)

    train_cls_model_max_combo()