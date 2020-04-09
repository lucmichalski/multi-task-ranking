

from retrieval.tools import Search, Eval, Pipeline
from static import Paths
import numpy as np

if __name__ == '__main__':


    results_dir = '/nfs/trec_car/data/entity_ranking/entity_parameter_tuning_3/'
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
    pipeline = Pipeline()

    pipeline.search_BM25_tune_parameter(index_path=Paths.entity_index,
                                        topics_path=Paths.entity_test_topics,
                                        qrels_path=Paths.entity_test_qrels,
                                        results_dir=results_dir,
                                        hits=100,
                                        b_list=[0.15],
                                        k1_list=[5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0])



