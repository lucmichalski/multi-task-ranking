

from retrieval.tools import Search, Eval

if __name__ == '__main__':
    passage_index_path = '/nfs/trec_car/index/anserini_paragraphs/lucene-index.car17v2.0.paragraphsv2'
    passage_test_topics_path = '/nfs/trec_car/data/bert_reranker_datasets/test.topics'
    passage_test_qrels_path = '/nfs/trec_car/data/bert_reranker_datasets/test.qrels'

    # entity_index_path = '/nfs/trec_car/index/anserini_pages/lucene-index.car17v2.0.pages.anserini.full_index.v1'
    # entity_test_topics_path = '/nfs/trec_car/data/entity_ranking/test_hierarchical.topics'
    # entity_test_qrels_path = '/nfs/trec_car/data/entity_ranking/test_hierarchical.qrels'

    run_path = '/nfs/trec_car/data/entity_ranking/test_hierarchical_passage_pyserini_10.run'
    hits = 10

    searcher_config = {
        'BM25': {'k1': 0.9, 'b': 0.4}
    }
    search = Search(index_path=passage_index_path, searcher_config=searcher_config)
    search.write_run_from_topics(topics_path=passage_test_topics_path, run_path=run_path, hits=hits)

    eval_config = {
        'map': {'k': None},
        'Rprec': {'k': None},
        'recip_rank': {'k': None},
        'P': {'k': 20},
        'recall': {'k': 40},
        'ndcg': {'k': 20},
    }
    eval = Eval()
    eval.write_eval_from_qrels_and_run(run_path=run_path, qrels_path=passage_test_qrels_path, eval_config=eval_config)


