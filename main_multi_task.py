
from multi_task.processing import MultiTaskDatasetByQuery
from multi_task.ranking import  train_cls_model_max_combo, train_cls_model, train_mutant_max_combo, train_mutant_multi_task_max_combo

if __name__ == '__main__':
    dir_path = '/nfs/trec_news_track/data/5_fold/'
    passage_model_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/exp/passage_ranking_bert_train_2epoch+8e-06lr/epoch1_batch2000/'
    entity_model_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/exp/entity_ranking_bert_train_2epoch+8e-06lr/epoch2_batch6000/'
    batch_size = 64*2
    max_rank = 100
    MultiTaskDatasetByQuery().build_dataset_by_query_entity_context_news(dir_path=dir_path,
                                                                        max_rank=max_rank,
                                                                        batch_size=batch_size,
                                                                        passage_model_path=passage_model_path,
                                                                        entity_model_path=entity_model_path
                                                                        )

    # train_mutant_max_combo()

    # batch_sizes = [64]
    # lrs = [0.000001, 0.00001, 0.0001, 0.0005, 0.001]
    # for batch_size in batch_sizes:
    #     for lr in lrs:
    #         train_mutant_multi_task_max_combo(batch_size=batch_size, lr=lr, mutant_type='mean')





