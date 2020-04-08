

from retrieval.tools import Search

if __name__ == '__main__':
    index_path = '/nfs/trec_car/index/anserini_pages/lucene-index.car17v2.0.pages.anserini.full_index.v1'
    topics_path = '/nfs/trec_car/data/entity_ranking/test_hierarchical.topics'
    run_path = '/nfs/trec_car/data/entity_ranking/test_hierarchical_10_pyserini_class.run'
    hits = 10
    searcher_settings = {
        'BM25':
            {
                'k1': 0.9,
                'b': 0.4
            }
    }
    search = Search(index_path=index_path, searcher_settings=searcher_settings)

    search.write_run_from_topics(topics_path=topics_path, run_path=run_path, hits=hits)