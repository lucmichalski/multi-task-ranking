
from collections import namedtuple

StaticPaths = namedtuple('StaticPaths', 'passage_index passage_test_topics passage_test_qrels entity_index entity_test_topics entity_test_qrels' )

Paths = StaticPaths(
    passage_index='/nfs/trec_car/index/anserini_paragraphs/lucene-index.car17v2.0.paragraphsv2',
    passage_test_topics='/nfs/trec_car/data/bert_reranker_datasets/test.topics',
    passage_test_qrels='/nfs/trec_car/data/bert_reranker_datasets/test.qrels',

    entity_index='/nfs/trec_car/index/anserini_pages/lucene-index.car17v2.0.pages.anserini.full_index.v1',
    entity_test_topics='/nfs/trec_car/data/entity_ranking/test_hierarchical.topics',
    entity_test_qrels='/nfs/trec_car/data/entity_ranking/test_hierarchical.qrels',
)

if __name__ == '__main__':
    print(Paths.passage_index)