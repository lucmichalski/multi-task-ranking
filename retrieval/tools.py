
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


class SearchTools:

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


    def process_query(self, q):
        """ Simple processing of query from TREC CAR format i.e. leave in utf-8 except space characters. """
        # Remove "enwiki:" from begging of string.
        i = 7
        assert q[:i] == "enwiki:"
        q = q[i:]
        # Add space for utf-8 space character.
        q = q.replace('%20', ' ')
        return q


    def decode_query(self, q, encoding='utf-8'):
        """ Process query using ut-8 decoding from TREC CAR format. """
        # Remove "enwiki:" from begging of string.
        i = 7
        assert q[:i] == "enwiki:"
        return urllib.parse.unquote(string=q[i:], encoding=encoding)


    def write_topics_from_qrels(self, qrels_path, topics_path=None):
        """ Given a TREC standard QRELS file in 'qrels_path', write TREC standard TOPICS file in 'file_path'. """
        # Build topics file path if None specified.
        if topics_path == None:
            # Assert ending of qrels path is ".qrels"
            i = len(".qrels")
            assert qrels_path[len(qrels_path)-i:] == ".qrels"
            topics_path = qrels_path[:len(qrels_path)-i] + '.topics'

        # Store queries already written to file.
        written_queries = []
        with open(topics_path, 'w') as topics_f:
            with open(qrels_path, 'r') as qrels_f:
                for line in qrels_f:
                    if "enwiki:" in line:
                        # Extract query from QRELS file.
                        print(line)
                        query, _, _, _ = line.split(' ')
                        if query not in written_queries:
                            # Write query to TOPICS file.
                            topics_f.write(query + '\n')
                            # Add query to 'written_queries' list.
                            written_queries.append(query)


    def combine_multiple_qrels(self, qrels_path_list, combined_qrels_path, combined_topics_path=None):
        """ Combines multiple qrels files into a single qrels file. """
        with open(combined_qrels_path, 'w') as f_combined_qrels:
            for qrels_path in qrels_path_list:
                print(qrels_path)
                with open(qrels_path, 'r') as f_qrels:
                    for line in f_qrels:
                        if "enwiki:" in line:
                            query, Q0, doc_id, rank = line.split(' ')
                            f_combined_qrels.write(" ".join((query, Q0, doc_id, rank)))
                        else:
                            print("Not valid query: {}")
                f_combined_qrels.write('\n')

        self.write_topics_from_qrels(qrels_path=combined_qrels_path, topics_path=combined_topics_path)


    def write_run_from_topics(self, topics_path, run_path, hits=10, printing_step=1000):
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

                    # Try to decode query correctly using URL utf-8 decoding. If this string causes an error within
                    # Pyserini's SimpleSearcher.search() use basic string processing only dealing with space characters.
                    try:
                        decoded_query = self.decode_query(q=query)
                        retrieved_hits = self.searcher.search(q=decoded_query, k=hits)
                    except ValueError:
                        print("URL utf-8 decoding did not work with Pyserini's SimpleSearcher.search()/JString: {}".format(query))
                        print("-> Using simple processing")
                        processed_query = self.process_query(q=query)
                        retrieved_hits = self.searcher.search(q=processed_query, k=hits)

                    for hit in retrieved_hits:
                        # Create and write run file.
                        run_line = " ".join((query, "Q0", hit.docid, str(rank), "{:.6f}".format(hit.score), "PYSERINI")) + '\n'
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


class EvalTools:

    """ Information retrieval eval class. """

    # Implemented eval metrics.
    def __init__(self):
        """ Map of metrics labels : calaulated functions """
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
            # Calculate discount cumulative gain.
            dcg = self.get_dcg(run=run, R=R, k=k)
            # Calculate perfect discount cumulative gain (i_dcg) from perfect run (i_run).
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
                if 'enwiki:' in query:
                    if query in qrels_dict:
                        qrels_dict[query].append(doc_id)
                    else:
                        qrels_dict[query] = [doc_id]
        return qrels_dict


    def get_query_metrics(self, run, R, eval_config):
        """ Build metrics (string and dict representation) from eval_config. """

        # Assert a valid eval metric config.
        assert isinstance(eval_config, dict)
        for k in eval_config.keys():
            assert k in self.implemented_metrics

        # Loop over implemented_metrics/eval_config.
        query_metrics = ''
        query_metrics_dict = {}
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
                query_metrics += metric_label + ' ' + "{:.6f}".format(metric) + ' '
                query_metrics_dict[metric_label] = metric

        return query_metrics, query_metrics_dict


    def write_eval_from_qrels_and_run(self, run_path, qrels_path, eval_config):
        """ Given qrels and run paths calculate evaluation metrics by query and aggreated and write to file. """
        # Eval path assumed to be run path with eval identifier added.
        eval_by_query_path = run_path + '.eval.by_query'
        # build qrels dict {query: [rel#1, rel#2, etc.]}.
        qrels_dict = self.get_qrels_dict(qrels_path=qrels_path)

        with open(eval_by_query_path, 'w') as f_eval_by_query:
            with open(run_path, 'r') as f_run:
                # Store query of run (i.e. previous query).
                topic_query = None
                # Store 'doc_id' for each hit in ascending order i.e. rank=1, rank=2, etc.
                run_doc_ids = []
                # Store binary relevance {0, 1} of each 'doc_id'.
                run = []
                for hit in f_run:
                    # Assumes run file is written in ascending order i.e. rank=1, rank=2, etc.
                    query, _, doc_id, _, _, _ = hit.split()

                    # If run batch complete.
                    if (topic_query != None) and (topic_query != query):
                        # Assert query of run in qrels
                        assert topic_query in qrels_dict
                        # For each doc_id find whether relevant (1) or not relevant (0), appending binary relevance to run.
                        for d in run_doc_ids:
                            if d in qrels_dict[topic_query]:
                                run.append(1)
                            else:
                                run.append(0)

                        # Calculate number of relevant docs in qrels (R).
                        R = len(qrels_dict[topic_query])
                        # Build query metric string.
                        query_metrics, _ = self.get_query_metrics(run=run, R=R, eval_config=eval_config)
                        # Write query metric string to file.
                        f_eval_by_query.write(topic_query + ' ' + query_metrics + '\n')
                        # Start next query.
                        run_doc_ids = []
                        run = []

                    # Add doc_id to run and map topic_query->query.
                    run_doc_ids.append(doc_id)
                    topic_query = query

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
                f_eval.write(k + ' ' + "{:.4f}".format(v) + '\n')

        return eval_metric


###############################################################################
############################# Pipeline Class ##################################
###############################################################################


class Pipeline:

    def search_BM25_tune_parameter(self, index_path, qrels_path, topics_path, results_dir, hits=10,
                                   b_list=np.arange(0.0, 1.1, 0.1), k1_list=np.arange(0.0, 3.2, 0.2)):
        """ Parameter search Pyserini BM25 algorithm (k1 & b parameters). """
        # Make results directory if does not exist
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)

        # Store parameter results.
        parameter_results = {}
        # Grid search of parameters.
        for b in b_list:
            for k1 in k1_list:

                # Run path for tune_parameter.
                run_path = os.path.join(results_dir, 'search_BM25_tune_parameter_k1={:.2f}_b={:2f}.run'.format(k1, b))

                # Search with evaluated parameters.
                searcher_config = {
                    'BM25': {'k1': k1, 'b': b}
                }
                search_tools = SearchTools(index_path=index_path, searcher_config=searcher_config)
                search_tools.write_run_from_topics(topics_path=topics_path, run_path=run_path, hits=hits)

                # Evaluate with evaluated runs.
                eval_config = {
                    'map': {'k': None},
                    'Rprec': {'k': None},
                    'recip_rank': {'k': None},
                    'P': {'k': 20},
                    'recall': {'k': 40},
                    'ndcg': {'k': 20},
                }
                eval = EvalTools()
                eval_metric = eval.write_eval_from_qrels_and_run(run_path=run_path, qrels_path=qrels_path, eval_config=eval_config)

                print('Parameters (k1: {} & b {}): {}'.format(k1, b, eval_metric))

                parameter_results['k1={:.2f}&b={:.2f}'.format(k1, b)] = eval_metric

        # Create aggregate DataFrame.
        df = pd.DataFrame(parameter_results).round(4)
        print('=====  Results Table =====')
        print(df)

        # Write aggregate DataFrame to csv.
        run_path = os.path.join(results_dir, 'results_df.csv')
        df.to_csv(run_path)



if __name__ == '__main__':
    print("hi")

    index_path = '/Users/iain/LocalStorage/anserini_index/car_entity_v9'
    topics_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'benchmarkY1_train_passage.topics')
    run_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'test_entity_1000.run')
    qrels_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'benchmarkY1_train_passage.qrels')
    results_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data')
    hits = 1000
    qrels_path_list = []
    files = ['fold-1-train.pages.cbor-hierarchical.qrels', 'fold-2-train.pages.cbor-hierarchical.qrels', 'fold-3-train.pages.cbor-hierarchical.qrels', 'fold-4-train.pages.cbor-hierarchical.qrels']
    for f in files:
        qrels_path_list.append(os.path.join(results_dir, f))
    print(qrels_path_list)
    searcher_config = {
        'BM25': {'k1': 5.5, 'b': 0.1}
    }
    search = SearchTools(index_path=index_path, searcher_config=searcher_config)
    search.combine_multiple_qrels(qrels_path_list=qrels_path_list, combined_qrels_path=qrels_path, combined_topics_path=topics_path)
    # search.write_topics_from_qrels(qrels_path=qrels_path)

    #search.write_run_from_topics(topics_path, run_path, hits)
    #search.write_topics_from_qrels(qrels_path=qrels_path)
    # eval_config = {
    #     'map': {'k': None},
    #     'Rprec': {'k': None},
    #     'recip_rank': {'k': None},
    #     'P': {'k': 20},
    #     'recall': {'k': 40},
    #     'ndcg': {'k': 20},
    # }
    #
    # eval = EvalTools()
    # eval.write_eval_from_qrels_and_run(run_path=run_path, qrels_path=qrels_path, eval_config=eval_config)
    # dev = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'benchmarkY1_dev_entity.qrels')
    #
    # search.write_topics_from_qrels(qrels_path=dev)
    # pipeline = Pipeline()
    #
    # pipeline.search_BM25_tune_parameter(index_path=index_path, topics_path=topics_path, qrels_path=qrels_path,
    #                                     results_dir=results_dir, hits=2, b_list=np.arange(0.0, 1.1, 0.5),
    #                                     k1_list=np.arange(0.0, 3.2, 1.6))


