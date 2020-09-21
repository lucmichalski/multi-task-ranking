
from multi_task.processing import MultiTaskDatasetByQuery

if __name__ == '__main__':
    MultiTaskDatasetByQuery().get_task_run_and_qrels(dataset='dev')