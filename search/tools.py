
from pyserini.search import pysearch
from pyserini.index import pyutils
from pyserini.analysis.pyanalysis import get_lucene_analyzer, Analyzer


def analyzer_string(string, stemming=True, stemmer='porter', stopwords=True):
    """ Build list of token using Lucene Analyzers which allows stemming and removal of stopwords. """
    analyzer = Analyzer(get_lucene_analyzer(stemming=stemming, stemmer=stemmer, stopwords=stopwords))
    return analyzer.analyze(string)


def get_contents_from_docid(index_path, doc_id):
    """ Retrieve raw document contents for from index. """
    index_utils = pyutils.IndexReaderUtils(index_path)
    return index_utils.get_raw_document_contents(docid=doc_id)


def search_simple(index_path, query, hits=10):
    """ Search query over index and return 'hits' number documents (docid, score)."""
    searcher = pysearch.SimpleSearcher(index_dir=index_path)
    return [(h.docid, h.score) for h in searcher.search(q=query, k=hits)]


def search_bm25(index_path, query,  hits=10, b=0.9, k1=0.5):
    """ Search query over index using BM25 and return 'hits' number documents (docid, score)."""
    searcher = pysearch.SimpleSearcher(index_dir=index_path)
    searcher.set_bm25_similarity(b=b, k1=k1)
    return [(h.docid, h.score) for h in searcher.search(q=query, k=hits)]


def search_bm25_with_rm3(index_path, query,  hits=10, b=0.9, k1=0.5, fb_terms=10, fb_docs=10, original_query_weight=0.5):
    """ Search query over index using BM25 with RM3 query expansion and return 'hits' number documents (docid, score)."""
    searcher = pysearch.SimpleSearcher(index_dir=index_path)
    searcher.set_bm25_similarity(b=b, k1=k1)
    searcher.set_rm3_reranker(fb_terms=fb_terms, fb_docs=fb_docs,original_query_weight=original_query_weight)
    return [(h.docid, h.score) for h in searcher.search(q=query, k=hits)]



if __name__ == '__main__':

    index_path = '/Users/iain/LocalStorage/anserini_index/car_entity_v9'
    query = 'Love  cats and dogs'
    hits = 10
    print(search_simple(index_path, query, hits))
    print(search_bm25(index_path, query, hits))
    print(search_bm25_with_rm3(index_path, query, hits))
    print(analyzer_string(string=query, stemming=True, stemmer='porter', stopwords=True))



