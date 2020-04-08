
from pyserini.search import pysearch
from pyserini.index import pyutils
from pyserini.analysis.pyanalysis import get_lucene_analyzer, Analyzer

import re
import urllib



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


def process_query(q):
    """ Process query from TREC CAR format. """
    # Remove "enwiki:" from begging of string.
    assert q[:7] == "enwiki:"
    q = q[7:]
    # Add spaces for special character.
    q = q.replace('%20', ' ')
    q = q.replace('/', ' ')
    return q


def decode_query(q, encoding='utf-8'):
    """ Process query using ut-8 decoding from TREC CAR format. """
    # Remove "enwiki:" from begging of string.
    assert q[:7] == "enwiki:"
    return urllib.parse.unquote(string=q[7:], encoding=encoding)


def write_run_file_from_topics(index_path, topics_path, run_path, hits, b=0.4, k1=0.9, printing_step=100):
    """ Write TREC RUN file using BM25. """
    print("Building searcher")
    searcher = pysearch.SimpleSearcher(index_dir=index_path)
    searcher.set_bm25_similarity(b=b, k1=k1)

    print("Beginning run.")
    print("-> Using topics: {}".format(topics_path))
    print("-> Create run file: {}".format(run_path))
    with open(topics_path, "r") as f_topics:
        with open(run_path, "w") as f_run:
            # Loop over topics.
            steps = 0
            for line in f_topics:
                rank = 1
                # Process query.
                query = line.split()[0]
                processed_query = process_query(q=query)
                for hit in searcher.search(q=processed_query, k=hits):
                    # Create and write run file.
                    run_line = " ".join((query, "Q0", hit.docid, str(rank), str(hit.score), "PYSERINI")) + '\n'
                    f_run.write(run_line)
                    # Next rank.
                    rank += 1
                steps += 1
                if (steps % printing_step == 0):
                    print("Processed query #{}: {}".format(steps, query))
    print("Completed run - written to run file: {}".format(run_path))



if __name__ == '__main__':

    import os

    index_path = '/Users/iain/LocalStorage/anserini_index/car_entity_v9'
    query = 'Love cats and dogs'
    topics_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'test.pages.cbor-hierarchical.entity.topics')
    run_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'test.pages.cbor-hierarchical.entity.run')
    hits = 10
    write_run_file_from_topics(index_path, topics_path, run_path, hits)



