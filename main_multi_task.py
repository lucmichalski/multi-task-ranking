
from multi_task.processing import MultiTaskDataset, create_extra_queries

if __name__ == '__main__':

    MultiTaskDataset().build_datasets(dir_path='/nfs/trec_car/data/entity_ranking/multi_task_data/')

    #create_extra_queries(dir_path='/nfs/trec_car/data/entity_ranking/multi_task_data/')