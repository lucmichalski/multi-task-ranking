
from multi_task.processing import MultiTaskDataset, create_extra_queries

if __name__ == '__main__':
    dir_path = '/nfs/trec_car/data/entity_ranking/multi_task_data/'

    #MultiTaskDataset().build_datasets(dir_path=dir_path)

    #create_extra_queries(dir_path='/nfs/trec_car/data/entity_ranking/multi_task_data/')

    MultiTaskDataset().cls_processing(dir_path=dir_path, batch_size=64*4)