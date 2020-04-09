
from pyserini.search import pysearch
from pyserini.index import pyutils
from pyserini.analysis.pyanalysis import get_lucene_analyzer, Analyzer

import pandas as pd
import numpy as np
import urllib
import math
import os


###############################################################################
############################## Search Class ###################################
###############################################################################


class Search:

    """ Information retrieval search class using Pyserini. """

    # Pyserini implemented searcher configs.
    implemented_searchers = ['BM25', 'BM25+RM3']


    def __init__(self, index_path, searcher_config=None):

        # Initialise absolute path to Anserini (Lucene) index.
        print("Index path: {}".format(index_path))
        self.index_path = index_path
        # Initialise index_utils for accessing index information.
        self.index_utils = pyutils.IndexReaderUtils(self.index_path)
        # Initialise searcher configuration with searcher_config dict. If no settings -> use SimpleSearcher.
        self.searcher = self.__build_searcher(searcher_config=searcher_config)


    def __build_searcher(self, searcher_config):
        """ Build Pyserini SimpleSearcher based on config."""
        searcher = pysearch.SimpleSearcher(index_dir=self.index_path)
        if isinstance(searcher_config, dict):
            # Check valid searcher config
            for k in searcher_config.keys():
                assert k in self.implemented_searchers

            # BM25 configuration.
            if 'BM25' in searcher_config:
                print('Using BM25 search model: {}'.format(searcher_config))
                searcher.set_bm25_similarity(b=searcher_config['BM25']['b'], k1=searcher_config['BM25']['k1'])
            # BM25 with RM3 query expansion configuration.
            elif 'BM25+RM3' in searcher_config:
                print('Using BM25+RM3 search model: {}'.format(searcher_config))
                searcher.set_bm25_similarity(b=searcher_config['BM25+RM3']['BM25']['b'],
                                             k1=searcher_config['BM25+RM3']['BM25']['k1'])
                searcher.set_rm3_reranker(fb_terms=searcher_config['BM25+RM3']['RM3']['fb_terms'],
                                          fb_docs=searcher_config['BM25+RM3']['RM3']['fb_docs'],
                                          original_query_weight=searcher_config['BM25+RM3']['RM3']['original_query_weight'])
            else:
                print('NOT VALID searcher_settings --> will use simple search: {}'.format(searcher_config))
                raise
            return searcher

        else:
            print('Using SimpleSearcher default')
            return searcher


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
                        run_line = " ".join((query, "Q0", hit.docid, str(rank), str(round(hit.score,6)), "PYSERINI")) + '\n'
                        f_run.write(run_line)
                        # Next rank.
                        rank += 1
                    steps += 1
                    if (steps % printing_step == 0):
                        print("Processed query #{}: {}".format(steps, query))
        print("Completed run - written to run file: {}".format(run_path))


###############################################################################
################################ Eval Class ###################################
###############################################################################


class Eval:

    """ Information retrieval eval class. """

    # Implemented eval metrics.
    def __init__(self):
        """  """
        self.implemented_metrics = {
            'map':          self.get_map,
            'Rprec':        self.get_Rprec,
            'recip_rank':   self.get_recip_rank,
            'P':            self.get_P,
            'recall':       self.get_recall,
            'ndcg':         self.get_ndcg
        }

    def get_map(self, run, R, k=None):
        """ Calculate mean average precision (MAP). """
        if k != None:
            run = run[:k]
        # Sum relevant docs in run.
        R_run = sum(run)
        # If relevant docs in run and total relevant docs above 0.
        if (R_run > 0.0) and (R > 0):
            # Initialise precision counter.
            precision_sum = 0
            # Loop of run and append precision to precision counter.
            for i, r in enumerate(run):
                if r == 1.0:
                    precision_sum += self.get_P(run=run, k=(i + 1))
            # Divide precision counter by total relevant docs.
            return precision_sum / R
        else:
            return 0.0


    def get_Rprec(self, run, R, k=None):
        """ Calculate R-precision. """
        if k != None:
            run = run[:k]
        if R > 0:
            # Reduce run at index #R and calculate precision.
            return sum(run[:R]) / R
        else:
            return 0.0


    def get_recip_rank(self, run, R=None, k=None):
        """ Calculate reciprocal rank. """
        if k != None:
            run = run[:k]
        for i, r in enumerate(run):
            # Return 1 / rank for first relevant.
            if r == 1.0:
                return 1 / (i + 1)
        return 0.0


    def get_P(self, run, R=None, k=20):
        """ Calculate precision (P) at kth rank. """
        if k != None:
            run = run[:k]
        return sum(run) / k


    def get_recall(self, run, R, k=40):
        """ Calculate recall at kth rank """
        if k != None:
            run = run[:k]
        R_run = sum(run)
        if R > 0:
            return R_run / R
        return 1.0


    def get_dcg(self, run, R=None, k=20):
        """ Calculate discount cumulative gain (DCG) at kth rank. """
        if k != None:
            run = run[:k]
        score = 0.0
        for order, rank in enumerate(run):
            score += float(rank) / math.log((order + 2))
        return score


    def get_ndcg(self, run, R, k=20):
        """ Calculate normalised discount cumulative gain (nDCG) at kth rank. """
        if R > 0:
            # Initialise discount cumulative gain.
            dcg = self.get_dcg(run=run, R=R, k=k)
            # Initialise perfect discount cumulative gain.
            i_run = [1] * R + [0] * (len(run) - R)
            i_dcg = self.get_dcg(run=i_run, R=R, k=k)
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


    def get_query_metrics(self, run, R, eval_config):
        """ Build metrics from eval_config. """

        # Assert a valid eval metric config.
        assert isinstance(eval_config, dict)
        for k in eval_config.keys():
            assert k in self.implemented_metrics

        # Loop over implemented_metrics/eval_config.
        query_metrics = ''
        for m in self.implemented_metrics:
            if m in eval_config:
                # Build label of metric.
                if eval_config[m]['k'] == None:
                    metric_label = m
                else:
                    metric_label = m + '_' + str(eval_config[m]['k'])
                # Calculate metric.
                metric = self.implemented_metrics[m](run=run, k=eval_config[m]['k'], R=R)
                # Append metric label and metric to string.
                query_metrics += metric_label + ' ' + str(round(metric, 4)) + ' '

        return query_metrics


    def write_eval_from_qrels_and_run(self, run_path, qrels_path, eval_config):
        """ TODO """
        # Eval path assumed to be run path with eval identifier added.
        eval_by_query_path = run_path + '.eval.by_query'
        # build qrels dict {query: [rel#1, rel#2, etc.]}.
        qrels_dict = self.get_qrels_dict(qrels_path=qrels_path)

        with open(eval_by_query_path, 'w') as f_eval_by_query:
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
                        # Build query metric string.
                        query_metrics = self.get_query_metrics(run=run, R=R, eval_config=eval_config)
                        # Write query metric string to file.
                        f_eval_by_query.write(query + ' ' + query_metrics + '\n')
                        # Start next query.
                        run_doc_ids = []
                        run = []

                    # Add doc_id to run and map run_query->query.
                    run_doc_ids.append(doc_id)
                    run_query = query

        # Store sum of metrics across all queries.
        eval_metric_sum = {}
        # Store number of queries - used to divide sum to find mean metric.
        query_counter = 0
        with open(eval_by_query_path, 'r') as f_eval_by_query:
            for query_metrics in f_eval_by_query:
                data = query_metrics.split()[1:]
                # For index and data item in query metric.
                for i, d in enumerate(data):
                    # If string of implemented_metrics is substring of data item --> update eval dict.
                    for m in self.implemented_metrics.keys():
                        if m in d:
                            if d not in eval_metric_sum:
                                eval_metric_sum[d] = 0
                            # Update
                            eval_metric_sum[d] += float(data[i+1])
                            break
                # Update counter.
                query_counter += 1

        # Find mean of metrics.
        eval_metric = {k: v / query_counter for k, v in eval_metric_sum.items()}

        # Write overall eval to file.
        eval_path = run_path + '.eval'
        with open(eval_path, 'w') as f_eval:
            for k, v in eval_metric.items():
                f_eval.write(k + ' ' + str(round(v, 4)) + '\n')

        return eval_metric


###############################################################################
############################# Pipeline Class ##################################
###############################################################################


class Pipeline:

    def search_BM25_tune_parameter(self, index_path, qrels_path, topics_path, results_dir, hits=10, b_list=np.arange(0.0, 1.1, 0.1),
                                   k1_list=np.arange(0.0, 3.2, 0.2)):
        """ """
        # Make results directory if does not exist
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)

        parameter_results = {}
        for k1 in k1_list:
            for b in b_list:

                # Run path for tune_parameter.
                run_path = os.path.join(results_dir, 'search_BM25_tune_parameter_k1={}_b={}.run'.format(round(k1, 2), round(b, 2)))

                # Search with evaluated parameters.
                searcher_config = {
                    'BM25': {'k1': k1, 'b': b}
                }
                search = Search(index_path=index_path, searcher_config=searcher_config)
                search.write_run_from_topics(topics_path=topics_path, run_path=run_path, hits=hits)

                # Evaluate with evaluated runs.
                eval_config = {
                    'map': {'k': None},
                    'Rprec': {'k': None},
                    'recip_rank': {'k': None},
                    'P': {'k': 20},
                    'recall': {'k': 40},
                    'ndcg': {'k': 20},
                }
                eval = Eval()
                eval_metric = eval.write_eval_from_qrels_and_run(run_path=run_path, qrels_path=qrels_path, eval_config=eval_config)
                print('k1: {} & b {}: {}'.format(k1, b, eval_metric))

                parameter_results['k1={}&b={}'.format(k1, b)] = eval_metric

        df = pd.DataFrame(parameter_results).round(4)
        print(df)
        run_path = os.path.join(results_dir, 'results_df.csv')
        df.to_csv(run_path)



if __name__ == '__main__':

    index_path = '/Users/iain/LocalStorage/anserini_index/car_entity_v9'
    query = 'Love cats and dogs'
    topics_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'test.pages.cbor-hierarchical.entity.topics')
    run_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'test.pages.cbor-hierarchical.entity.run')
    qrels_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'test.pages.cbor-hierarchical.entity.qrels')
    results_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'results')
    hits = 10

    # search = Search()
    # search.write_run_from_topics(index_path, topics_path, run_path, hits)
    # eval_config = {
    #     'map': {'k': None},
    #     'Rprec': {'k': None},
    #     'recip_rank': {'k': None},
    #     'P': {'k': 20},
    #     'recall': {'k': 40},
    #     'ndcg': {'k': 20},
    # }
    #
    # eval = Eval()
    # eval.write_eval_from_qrels_and_run(run_path=run_path, qrels_path=qrels_path, eval_config=eval_config)

    pipeline = Pipeline()

    pipeline.search_BM25_tune_parameter(index_path=index_path, topics_path=topics_path, results_dir=results_dir,
                                        hits=2, b_list=np.arange(0.0, 1.1, 0.5), k1_list=np.arange(0.0, 3.2, 1.4))


