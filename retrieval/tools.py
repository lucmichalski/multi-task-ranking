
from pyserini.search import pysearch
from pyserini.index import pyutils
from pyserini.analysis.pyanalysis import get_lucene_analyzer, Analyzer

import numpy as np
import urllib
import re
import os



class Search:
    """  """

    implemented_searchers = ['BM25', 'BM25+RM3']

    def __init__(self, index_path, searcher_settings=None):

        # Initialise absolute path to Anserini (Lucene) index.
        self.index_path = index_path
        # Initialise index_utils for accessing index information.
        self.index_utils = pyutils.IndexReaderUtils(self.index_path)
        # Initialise searcher configuration with searcher_settings dict. If no settings -> use SimpleSearcher.
        self.searcher = pysearch.SimpleSearcher(index_dir=self.index_path)
        if isinstance(searcher_settings, dict):
            # BM25 configuration.
            if 'BM25' in searcher_settings:
                self.searcher.set_bm25_similarity(b=searcher_settings['BM25']['b'], k1=searcher_settings['BM25']['k1'])
            # BM25 with RM3 query expansion configuration.
            elif 'BM25+RM3' in searcher_settings:
                self.searcher.set_bm25_similarity(b=searcher_settings['BM25+RM3']['BM25']['b'],
                                                  k1=searcher_settings['BM25+RM3']['BM25']['k1'])
                self.searcher.set_rm3_reranker(fb_terms=searcher_settings['BM25+RM3']['RM3']['fb_terms'],
                                               fb_docs=searcher_settings['BM25+RM3']['RM3']['fb_docs'],
                                               original_query_weight=searcher_settings['BM25+RM3']['RM3']['original_query_weight'])
            else:
                print('NOT VALID searcher_settings --> will use simple search')
                raise


    def analyzer_string(self, string, stemming=True, stemmer='porter', stopwords=True):
        """ Build list of token using Lucene Analyzers which allows stemming and removal of stopwords. """
        analyzer = Analyzer(get_lucene_analyzer(stemming=stemming, stemmer=stemmer, stopwords=stopwords))
        return analyzer.analyze(string)


    def get_contents_from_docid(self, doc_id):
        """ Retrieve raw document contents for from index. """
        return self.index_utils.get_raw_document_contents(docid=doc_id)


    def __get_ranking_from_searcher(self, query, hits):
        """ Given Pyserini searcher and query return ranking i.e. list of (docid, score) tuples. """
        return [(h.docid, h.score) for h in self.searcher.search(q=query, k=hits)]


    def search(self, query, hits=10):
        """ Search query over index and return 'hits' number documents (docid, score)."""
        return self.__get_ranking_from_searcher(query=query, hits=hits)


    def __process_query(self, q):
        """ Process query from TREC CAR format. """
        # Remove "enwiki:" from begging of string.
        assert q[:7] == "enwiki:"
        q = q[7:]
        # Add spaces for special character.
        q = q.replace('%20', ' ')
        q = q.replace('/', ' ')
        return q


    def __decode_query(self, q, encoding='utf-8'):
        """ Process query using ut-8 decoding from TREC CAR format. """
        # Remove "enwiki:" from begging of string.
        assert q[:7] == "enwiki:"
        return urllib.parse.unquote(string=q[7:], encoding=encoding)


    def write_run_from_topics(self, topics_path, run_path, hits, printing_step=100):
        """ Write TREC RUN file using BM25 from topics file. """
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
                    processed_query = self.__process_query(q=query)
                    for hit in self.searcher.search(q=processed_query, k=hits):
                        # Create and write run file.
                        run_line = " ".join((query, "Q0", hit.docid, str(rank), str(hit.score), "PYSERINI")) + '\n'
                        f_run.write(run_line)
                        # Next rank.
                        rank += 1
                    steps += 1
                    if (steps % printing_step == 0):
                        print("Processed query #{}: {}".format(steps, query))
        print("Completed run - written to run file: {}".format(run_path))


class Eval:

    def __init__(self):
        """  """
        self.implemented_metrics = {
            'map':          self.get_map,
            'r_prec':       self.get_R_prec,
            'recip_rank':   self.get_recip_rank,
            'precision':    self.get_precision,
            'recall':       self.get_recall,
            'ndcg':         self.get_ndcg
        }

    def get_map(self, run, R):
        """ Calculate mean average precision (MAP). """
        # Sum relevant docs in run.
        R_run = sum(run)
        # If relevant docs in run and total relevant docs above 0.
        if (R_run > 0.0) and (R > 0):
            # Initialise precision counter.
            precision_sum = 0
            # Loop of run and append precision to precision counter.
            for i, r in enumerate(run):
                if r == 1.0:
                    precision_sum += self.get_precision(run=run, k=(i + 1))
            # Divide precision counter by total relevant docs.
            return precision_sum / R
        else:
            return 0.0


    def get_R_prec(self, run, R):
        """ Calculate R-precision. """
        if R > 0:
            # Reduce run at index #R and calculate precision.
            return sum(run[:R]) / R
        else:
            return 0.0


    def get_recip_rank(self, run):
        """ Calculate reciprocal rank. """
        for i, r in enumerate(run):
            # Return 1 / rank for first relevant.
            if r == 1.0:
                return 1 / (i + 1)
        return 0.0


    def get_precision(self, run, k=20):
        """ Calculate precision at kth rank. """
        run_k = run[:k]
        return sum(run_k) / k


    def get_recall(self, run, R, k=40):
        """ Calculate recall at kth rank """
        run_k = run[:k]
        R_run = sum(run_k)
        if R > 0:
            return R_run / R
        return 1.0


    def get_ndcg(self, run, R, k=20):
        """ Calculate normalised discount cumulative gain (NDCG) at kth rank. """
        run_k = run[:k]
        # Initialise discount cumulative gain.
        dcg = 0
        # Initialise perfect discount cumulative gain.
        i_dcg = 0
        R_run = sum(run)
        if (R_run > 0) and (R > 0):
            for i, r in enumerate(run_k):
                if i == 0:
                    if (i + 1) <= R:
                        i_dcg += 1
                    dcg += r
                else:
                    discount = np.log2(i + 2)
                    if (i + 1) <= R:
                        i_dcg += 1 / discount
                    dcg += r / discount
            # Normalise cumulative gain by dividing 'discount cumulative gain' by 'perfect discount cumulative gain'.
            return dcg / i_dcg
        else:
            return 0.0


    def get_qrels_dict(self, qrels_path):
        """ Build a dictionary from a qrels file: {query: [rel#1, rel#2, rel#3, ...]}. """
        qrels_dict = {}
        with open(qrels_path) as qrels_file:
            # Read each line of qrels file.
            for line in qrels_file:
                query, _, doc_id, _ = line.strip().split(" ")
                # key: query, value: list of doc_ids
                if query in qrels_dict:
                    qrels_dict[query].append(doc_id)
                else:
                    qrels_dict[query] = [doc_id]
        return qrels_dict


    def get_query_metrics(self, run, R):
        run_metrics = []
        run_metrics.append(self.get_map(run=run, R=R))
        run_metrics.append(self.get_R_prec(run=run, R=R))
        run_metrics.append(self.get_recip_rank(run=run))
        run_metrics.append(self.get_precision(run=run, k=20))
        run_metrics.append(self.get_recall(run=run, k=40, R=R))
        run_metrics.append(self.get_ndcg(run=run, R=R, k=20))

        return run_metrics

    def write_eval_from_qrels_and_run(self, run_path, qrels_path):
        """ """
        # Eval path assumed to be run path with eval identifier added.
        eval_path = run_path + '.eval.by_query'
        #
        qrels_dict = self.get_qrels_dict(qrels_path=qrels_path)

        with open(eval_path, 'w') as f_eval:
            with open(run_path, 'r') as f_run:
                # Store query of run (i.e. previous query).
                run_query = None
                # Store 'doc_id' for each hit in ascending order i.e. rank=1, rank=2, etc.
                run_doc_ids = []
                # Store binary relevance {0, 1} of each 'doc_id'.
                run = []
                for hit in f_run:
                    # Assumes run file is written in ascending order i.e. rank=1, rank=2, etc.
                    query, _, doc_id, _, _, _ = hit.split()

                    # If run batch complete.
                    if (run_query != None) and (run_query != query):
                        # Assert query of run in qrels
                        assert run_query in qrels_dict
                        # For each doc_id find whether relevant (1) or not relevant (0), appending binary relevance to run.
                        for d in run_doc_ids:
                            if d in qrels_dict[run_query]:
                                run.append(1)
                            else:
                                run.append(0)
                        # Calculate number of relevant docs in qrels (R).
                        R = len(qrels_dict[run_query])
                        metrics = self.get_query_metrics(run=run, R=R)
                        f_eval.write(str(metrics) + '\n')
                        run_doc_ids = []
                        run = []

                    run_doc_ids.append(doc_id)
                    run_query = query
                    print(query, doc_id)


if __name__ == '__main__':

    index_path = '/Users/iain/LocalStorage/anserini_index/car_entity_v9'
    query = 'Love cats and dogs'
    topics_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'test.pages.cbor-hierarchical.entity.topics')
    run_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'test.pages.cbor-hierarchical.entity.run')
    qrels_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'test.pages.cbor-hierarchical.entity.qrels')
    hits = 10

    # search = Search()
    # search.write_run_from_topics(index_path, topics_path, run_path, hits)

    # eval = Eval()
    # eval.write_eval_from_qrels_and_run(run_path, qrels_path)


