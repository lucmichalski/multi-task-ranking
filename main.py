

from search.tools import write_run_file_from_topics

if __name__ == '__main__':
    index_path = '/nfs/trec_car/index/anserini_pages/lucene-index.car17v2.0.pages.anserini.full_index.v1'
    topics_path = '/nfs/trec_car/data/entity_ranking/test_hierarchical.topics'
    run_path = '/nfs/trec_car/data/entity_ranking/test_hierarchical_10_pyserini_lightweight_processing_new_params.run'
    hits = 10
    write_run_file_from_topics(index_path, topics_path, run_path, hits)