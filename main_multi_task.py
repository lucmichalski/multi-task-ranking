
from multi_task.processing import MultiTaskDatasetByQuery

if __name__ == '__main__':
    dir_path = '/nfs/trec_car/data/entity_ranking/multi_task_data_by_query/'
    MultiTaskDatasetByQuery().build_dataset_by_query(dir_path=dir_path)