
from collections import namedtuple

StaticPaths = namedtuple('StaticPaths', 'corpus index test_topics test_qrels')

CarPassagePaths = StaticPaths(
    corpus='/nfs/trec_car/data/paragraphs/dedup.articles-paragraphs.cbor',
    index='/nfs/trec_car/index/anserini_paragraphs/lucene-index.car17v2.0.paragraphsv2',
    test_topics='/nfs/trec_car/data/bert_reranker_datasets/test.topics',
    test_qrels='/nfs/trec_car/data/bert_reranker_datasets/test.qrels',
)

CarEntityPaths = StaticPaths(
    corpus='/nfs/trec_car/data/pages/unprocessedAllButBenchmark.Y2.cbor',
    index='/nfs/trec_car/index/anserini_pages/lucene-index.car17v2.0.pages.anserini.full_index.v1',
    test_topics='/nfs/trec_car/data/entity_ranking/test_hierarchical_entity.topics',
    test_qrels='/nfs/trec_car/data/entity_ranking/test_hierarchical_entity.qrels',
)

NewsPassagePaths = StaticPaths(
    corpus='/nfs/trec_news_track/WashingtonPost.v2.tar.gz',
    index='/nfs/trec_news_track/index/WashingtonPost.v2.index.anserini.v1',
    test_topics='TODO',
    test_qrels='TODO',
)

NewsEntityPaths = StaticPaths(
    corpus='TODO',
    index='TODO',
    test_topics='TODO',
    test_qrels='TODO',
)

if __name__ == '__main__':
    print(CarPassagePaths)