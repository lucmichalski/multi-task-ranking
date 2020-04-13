

from retrieval.tools import SearchTools, EvalTools, Pipeline
from retrieval.dataset_processing import TrecCarProcessing
from metadata import EntityPaths, PassagePaths
import numpy as np
from transformers import BertTokenizer


if __name__ == '__main__':


    # results_dir = '/nfs/trec_car/data/entity_ranking/entity_parameter_tuning_4/'
    #
    # run_path = '/nfs/trec_car/data/entity_ranking/test_hierarchical_passage_pyserini_10.run'
    # hits = 1000
    #
    # searcher_config = {
    #     'BM25': {'k1': 0.9, 'b': 0.4}
    # }
    # search = Search(index_path=Paths.passage_index, searcher_config=searcher_config)
    # search.write_run_from_topics(topics_path=Paths.passage_test_topics, run_path=run_path, hits=hits)
    #
    # eval_config = {
    #     'map': {'k': None},
    #     'Rprec': {'k': None},
    #     'recip_rank': {'k': None},
    #     'P': {'k': 20},
    #     'recall': {'k': 40},
    #     'ndcg': {'k': 20},
    # }
    # eval = Eval()
    # eval.write_eval_from_qrels_and_run(run_path=run_path, qrels_path=passage_test_qrels_path, eval_config=eval_config)
    #
    # pipeline = Pipeline()
    #
    # pipeline.search_BM25_tune_parameter(index_path=EntityPaths.entity_index,
    #                                     topics_path=EntityPaths.entity_test_topics,
    #                                     qrels_path=EntityPaths.entity_test_qrels,
    #                                     results_dir=results_dir,
    #                                     hits=1000,
    #                                     b_list=[0.1, 0.15, 0.2, 0.25],
    #                                     k1_list=[4.5, 5.5, 6.5, 7.5, 8.5])

    index_path = '/nfs/trec_car/index/anserini_paragraphs/lucene-index.car17v2.0.paragraphsv2'
    run_path = '/nfs/trec_car/data/bert_reranker_datasets/test_10.run'
    qrels_path = '/nfs/trec_car/data/bert_reranker_datasets/test_10.qrels'
    data_dir_path = '/nfs/trec_car/data/bert_reranker_datasets/'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 512
    processing = TrecCarProcessing(qrels_path=qrels_path,
                                   run_path=run_path,
                                   index_path=index_path,
                                   data_dir_path=data_dir_path,
                                   tokenizer=tokenizer,
                                   max_length=max_length)
    processing.build_dataset(sequential=True)




