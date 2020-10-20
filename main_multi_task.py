
from multi_task.processing import MultiTaskDatasetByQuery
from multi_task.ranking import train_cls_model_max_combo, train_cls_model, train_mutant_max_combo, \
    train_mutant_multi_task_max_combo, train_mutant_multi_task_max_combo_news
from multi_task.mutant import train_and_dev_mutant, get_dev_dataset, get_train_dataset

if __name__ == '__main__':
    # dir_path = '/nfs/trec_news_track/data/5_fold/'
    # passage_model_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/exp/passage_ranking_bert_train_2epoch+8e-06lr/epoch1_batch2000/'
    # entity_model_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/exp/entity_ranking_bert_train_2epoch+8e-06lr/epoch2_batch6000/'
    # batch_size = 64*2
    # max_rank = 100
    # MultiTaskDatasetByQuery().build_dataset_by_query_entity_context_news(dir_path=dir_path,
    #                                                                     max_rank=max_rank,
    #                                                                     batch_size=batch_size,
    #                                                                     passage_model_path=passage_model_path,
    #                                                                     entity_model_path=entity_model_path
    #                                                                     )

    # train_mutant_max_combo()

    # batch_sizes = [64]
    # lrs = [0.000001, 0.00001, 0.0001, 0.0005, 0.001]
    # for batch_size in [32]:
    #     for lr in [0.001, 0.0001]:
    #         train_mutant_multi_task_max_combo_news(batch_size=batch_size, lr=lr, mutant_type='mean', keyword=True)
    # for batch_size in [32]:
    #     for lr in [0.001, 0.0001]:
    #         train_mutant_multi_task_max_combo_news(batch_size=batch_size, lr=lr, mutant_type='max', keyword=True)
    #
    #
    #
    #
    #train_dir_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/mutant_data/train/'
    #dev_dir_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/mutant_data/valid/'
    doc_to_entity_map_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/doc_to_entity_map.json'
    #file_name = '_mutant_max.json'
    #dev_qrels_path = '/nfs/trec_news_track/data/5_fold/scaled_5fold_0_data/passage_valid.qrels'

    train_dir_path = '/nfs/trec_car/data/entity_ranking/multi_task_data_by_query_1000/train_data/'
    dev_dir_path = '/nfs/trec_car/data/entity_ranking/multi_task_data_by_query/dev_data/'
    file_name = 'data_bi_encode_ranker_entity_context.json'
    dev_qrels_path = '/nfs/trec_car/data/entity_ranking/benchmarkY1_hierarchical_passage_train_data/benchmarkY1_train_passage.qrels'

    dev_save_path_run = dev_dir_path + 'mutant_dev.run_data'
    dev_save_path_dataset = dev_dir_path + 'mutant_dev.pt'
    train_save_path_dataset = train_dir_path + 'mutant_train.pt'

    get_dev_dataset(save_path_dataset=dev_save_path_dataset, save_path_run=dev_save_path_run, dir_path=dev_dir_path, doc_to_entity_map_path=doc_to_entity_map_path, file_name=file_name, max_seq_len=16)
    get_train_dataset(save_path_dataset=train_save_path_dataset, dir_path=train_dir_path, doc_to_entity_map_path=doc_to_entity_map_path, file_name=file_name, max_seq_len=16)

    train_and_dev_mutant(dev_save_path_run=dev_save_path_run, dev_save_path_dataset=dev_save_path_dataset, dev_qrels_path=dev_qrels_path, train_save_path_dataset=train_save_path_dataset, lr=0.0001, epoch=5, max_seq_len=16, batch_size=32)