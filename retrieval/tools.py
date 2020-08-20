
from pyserini.search import pysearch
from pyserini.index import pyutils
from pyserini.analysis.pyanalysis import get_lucene_analyzer, Analyzer
from metadata import NewsPassagePaths, CarEntityPaths

from collections import Counter
import pandas as pd
import numpy as np
import random
import urllib
import shutil
import math
import json
import os
import re

###############################################################################
######################### Retrieval Utils Class ###############################
###############################################################################

class RetrievalUtils:

    query_start_stings = ['enwiki:', 'tqa:']


    def get_qrels_binary_dict(self, qrels_path):
        """ Build a dictionary from a qrels file with binary relevance: {query: [rel#1, rel#2, rel#3, ...]}. """
        if isinstance(qrels_path, str):
            qrels_dict = {}
            with open(qrels_path, 'r', encoding="utf-8") as qrels_file:
                # Read each line of qrels file.
                for line in qrels_file:
                    if len(line) > 4:
                        query, _, doc_id, score = self.unpack_qrels_line(line)
                        if query not in qrels_dict:
                            qrels_dict[query] = []
                        if float(score) > 0.0:
                            # key: query, value: list of doc_ids
                            qrels_dict[query].append(doc_id)
            return qrels_dict
        else:
            return None


    def get_qrels_norm_dict(self, qrels_path):
        """ Build a dictionary from a qrels file with normalised relevance: {query: {rel#1: score},... }. """
        if isinstance(qrels_path, str):
            # Get max score form qrels file to normalise with.
            with open(qrels_path, 'r', encoding="utf-8") as qrels_file:
                max_score = 0.0
                # Read each line of qrels file.
                for line in qrels_file:
                    if len(line) > 4:
                        _, _, _, score = self.unpack_qrels_line(line)
                        if float(score) > max_score:
                            max_score = float(score)

            qrels_dict = {}
            with open(qrels_path, 'r', encoding="utf-8") as qrels_file:
                # Read each line of qrels file.
                for line in qrels_file:
                    if len(line) > 4:
                        query, _, doc_id, score = self.unpack_qrels_line(line)
                        if query not in qrels_dict:
                            qrels_dict[query] = {}
                        if float(score) > 0.0:
                            # key: query, value: list of doc_ids
                            qrels_dict[query][doc_id] = float(score) / max_score

            return qrels_dict
        else:
            return None

    def get_qrels_scaled_dict(self, qrels_path):
        """ Build a dictionary from a qrels file with normalised relevance: {query: {rel#1: score},... }. """
        if isinstance(qrels_path, str):
            qrels_dict = {}
            with open(qrels_path, 'r', encoding="utf-8") as qrels_file:
                # Read each line of qrels file.
                for line in qrels_file:
                    if len(line) > 4:
                        query, _, doc_id, score = self.unpack_qrels_line(line)
                        if query not in qrels_dict:
                            qrels_dict[query] = {}
                        if float(score) >= 0.0:
                            # key: query, value: list of doc_ids
                            qrels_dict[query][doc_id] = float(score)

            return qrels_dict
        else:
            return None

    def test_valid_line_car(self, line):
        """ Return bool whether valid starting substring is in line"""
        return any(substring in line for substring in self.query_start_stings)


    def unpack_run_line(self, line):
        """ Unpack line from trec run file. """
        split_line = line.strip().split()
        query = split_line[0]
        q = split_line[1]
        doc_id = split_line[2]
        rank = split_line[3]
        score = split_line[4]
        name = split_line[5]
        return query, q, doc_id, rank, score, name


    def unpack_qrels_line(self, line):
        """ Unpack line from trec qrels file. """
        split_line = line.strip().split()
        query = split_line[0]
        q = split_line[1]
        doc_id = split_line[2]
        score = split_line[3]
        return query, q, doc_id, score


###############################################################################
############################## Search Class ###################################
###############################################################################

default_searcher_config = {
        'BM25': {'k1': 0.9, 'b': 0.4}
    }

class SearchTools:

    """ Information retrieval search class using Pyserini. """

    # Pyserini implemented searcher configs.
    implemented_searchers = ['BM25', 'BM25+RM3']
    retrieval_utils = RetrievalUtils()

    def __init__(self, index_path=None, searcher_config=default_searcher_config):
        # Initialise absolute path to Anserini (Lucene) index.
        print("Index path: {}".format(index_path))
        self.index_path = index_path
        # Initialise index_utils for accessing index information.
        self.index_utils = self.__build_index_utils(index_path=index_path)
        # Initialise searcher configuration with searcher_config dict. If no settings -> use SimpleSearcher.
        self.searcher = self.__build_searcher(searcher_config=searcher_config, index_path=index_path)


    def __build_index_utils(self, index_path):
        """ Initialise index_utils for accessing index information. """
        if index_path == None:
            return None
        else:
            return pyutils.IndexReaderUtils(self.index_path)


    def __build_searcher(self, searcher_config, index_path):
        """ Build Pyserini SimpleSearcher based on config."""
        if searcher_config == None or index_path == None:
            return None
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


    def __remove_query_start_car(self, q):
        """ Removes beginning of query. """
        for start_str in self.retrieval_utils.query_start_stings:
            i = len(start_str)
            if start_str == q[:i]:
                return q[i:]


    def process_query_car(self, q):
        """ Simple processing of query from TREC CAR format i.e. leave in utf-8 except space characters. """
        # Remove begging of string.
        q = self.__remove_query_start_car(q=q)
        # Add space for utf-8 space character.
        q = q.replace('%20', ' ')
        return q


    def decode_query_car(self, q, encoding='utf-8'):
        """ Process query using ut-8 decoding from TREC CAR format. """
        # Remove "enwiki:" from begging of string.
        q = self.__remove_query_start_car(q=q)
        return urllib.parse.unquote(string=q, encoding=encoding)


    def write_topics_car(self, qrels_path, topics_path=None):
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
                    if self.retrieval_utils.test_valid_line(line=line):
                        # Extract query from QRELS file.
                        query, _, _, _ = self.retrieval_utils.unpack_qrels_line(line=line)
                        if query not in written_queries:
                            # Write query to TOPICS file.
                            topics_f.write(query + '\n')
                            # Add query to 'written_queries' list.
                            written_queries.append(query)


    def write_topics_news(self, xml_topics_path, topics_path):
        """ Write TREC News Track topics from XLM topics file. """
        with open(topics_path, 'w') as f_out:
            with open(xml_topics_path, 'r') as f_in:
                for line in f_in:
                    if 'docid' in line:
                        start_i = [m.span() for m in re.finditer('<docid>', line)][0][1]
                        end_i = [m.span() for m in re.finditer('</docid>', line)][0][0]
                        i = str(line[start_i:end_i])
                        f_out.write(i + '\n')


    def get_news_ids_maps(self, xml_topics_path, ranking_type='passage'):
        """ Build dict map from intermediate ids to Washington Post ids {intermediate_id: passage_id} """
        passage_id_map = {}
        entity_id_map = {}
        with open(xml_topics_path, 'r') as f:
            for line in f:
                # Passage intermediate_id
                if '<num>' in line:
                    start_i = [m.span() for m in re.finditer('<num> Number: ', line)][0][1]
                    end_i = [m.span() for m in re.finditer(' </num>', line)][0][0]
                    passage_temp_id = line[start_i:end_i]
                # Passage id
                if '<docid>' in line:
                    start_i = [m.span() for m in re.finditer('<docid>', line)][0][1]
                    end_i = [m.span() for m in re.finditer('</docid>', line)][0][0]
                    passage_id = line[start_i:end_i]
                    passage_id_map[passage_temp_id] = passage_id

                if ranking_type == 'entity':
                    # Entity intermediate_id
                    if '<id>' in line:
                        start_i = [m.span() for m in re.finditer('<id> ', line)][0][1]
                        end_i = [m.span() for m in re.finditer(' </id>', line)][0][0]
                        entity_temp_id = line[start_i:end_i]
                    # Entity id
                    if '<link>' in line:
                        start_i = [m.span() for m in re.finditer('<link>', line)][0][1]
                        end_i = [m.span() for m in re.finditer('</link>', line)][0][0]
                        entity_id = line[start_i:end_i]
                        entity_id_map[entity_temp_id] = entity_id

        return passage_id_map, entity_id_map


    def write_qrels_news(self, xml_topics_path, old_qrels_path, qrels_path, ranking_type='passage'):
        """ Write qrels for News Track augmenting intermediate query ids. """
        assert ranking_type == 'passage' or ranking_type == 'entity'
        # Build maps - intermediate_id: id
        passage_id_map, entity_id_map = self.get_news_ids_maps(xml_topics_path=xml_topics_path, ranking_type=ranking_type)

        # Augment qrels with true query_ids.
        with open(qrels_path, 'w') as f_out:
            with open(old_qrels_path, 'r') as f_in:
                for line in f_in:
                    # Passage qrels.
                    if ranking_type == 'passage':
                        query_temp, q, doc_id, score = self.retrieval_utils.unpack_qrels_line(line)
                        query = passage_id_map[query_temp]
                    # Entity qrels.
                    else:
                        query_temp, q, doc_id_temp, score = self.retrieval_utils.unpack_qrels_line(line)
                        query = passage_id_map[query_temp]
                        doc_id = entity_id_map[doc_id_temp]
                    f_out.write(" ".join((query, q, doc_id, str(score))) + '\n')


    def combine_multiple_qrels(self, qrels_path_list, combined_qrels_path):
        """ Combines multiple qrels files into a single qrels file. """
        with open(combined_qrels_path, 'w') as f_combined_qrels:
            for qrels_path in qrels_path_list:
                print(qrels_path)
                with open(qrels_path, 'r') as f_qrels:
                    for line in f_qrels:
                        if self.retrieval_utils.test_valid_line(line=line):
                            f_combined_qrels.write(line)
                        else:
                            print("Not valid query: {}")
                f_combined_qrels.write('\n')


    def write_tree_no_root_qrels_from_tree_qrels(self, tree_qrels_path, tree_no_root_qrels_path):
        """ Write tree-no-root qrels from tree qrels file (remove queries that do not contain '/') """
        with open(tree_no_root_qrels_path, 'w') as f_tree_no_root_qrels:
            with open(tree_qrels_path, 'r') as f_tree_qrels:
                for line in f_tree_qrels:
                    if '/' in line:
                        f_tree_no_root_qrels.write(line)


    def write_new_qrels_with_mapped_scores(self, old_qrels_path, new_qrels_path, mapped_scores):
        """ Augment qrels file mapping qrels scores to different scores. """
        with open(new_qrels_path, 'w') as f_new:
            with open(old_qrels_path, 'r') as f_old:
                for line in f_old:
                    if self.retrieval_utils.test_valid_line_car(line):
                        query, q, doc_id, old_score = self.retrieval_utils.unpack_qrels_line(line=line)
                        assert int(old_score) in mapped_scores, "score: {} not in mapped_scores: {}".format(old_score, mapped_scores)
                        new_score = mapped_scores[int(old_score)]
                        if new_score != None:
                            f_new.write(" ".join((query, q, doc_id, str(new_score))) + '\n')


    def write_run_car(self, topics_path, run_path, hits=10, printing_step=1000):
        """ Write TREC RUN file using BM25 from topics file. """
        print("Beginning run.")
        print("-> Using topics: {}".format(topics_path))
        print("-> Create run file: {}".format(run_path))
        with open(topics_path, "r") as f_topics:
            with open(run_path, "w") as f_run:
                # Loop over topics.
                steps = 0
                for query in f_topics:
                    query = query.rstrip()
                    rank = 1
                    # Try to decode query correctly using URL utf-8 decoding. If this string causes an error within
                    # Pyserini's SimpleSearcher.search() use basic string processing only dealing with space characters.
                    try:
                        decoded_query = self.decode_query_car(q=query)
                        retrieved_hits = self.searcher.search(q=decoded_query, k=hits)
                    except ValueError:
                        print("URL utf-8 decoding did not work with Pyserini's SimpleSearcher.search()/JString: {}".format(query))
                        print("-> Using simple processing")
                        processed_query = self.process_query_car(q=query)
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


    def process_query_news(self, query_dict, query_type):
        """ Process news query from dict of Washington Post article. """
        assert query_type == 'title' or query_type == 'title+contents'
        # Simply return title string.
        if query_type == 'title':
            return query_dict['title']
        # Append contents text to title.
        elif query_type == 'title+contents':
            # Extract title
            try:
                title = query_dict['title']
            except:
                title = ""
                print('FAILED TO PARSE TITLE')
            # Extract conents text
            content_text = ""
            for content in query_dict['contents']:
                try:
                    if 'content' in content.keys():
                        if isinstance(content['content'], dict) == False:
                            rule = r'<a href=.*\>|<a class=.*\>|<span class=.*\>|</span>|<span>|</span>|<i>|</i>|<strong>|</strong>|<b>|</b>|<br />'
                            text = re.sub(rule, '', str(content['content']))
                            content_text += " " + str(text)
                except:
                    print('FAILED TO PARSE CONTENTS')
            # Title + contents.
            if isinstance(title, str):
                news_query = title + ' ' + content_text
            else:
                news_query = content_text

            return news_query


    def write_entity_run_news(self, run_path, qrels_path, query_type, words=100, hits=250000,
                              news_index_path=NewsPassagePaths.index):
        """ Write PySerini BM25 for TREC News Track - query_type {'passage', 'entity}. """
        assert query_type == 'title' or query_type == 'title+contents'

        # Initialise SearchTools for news index (queries)
        search_tools_news = SearchTools(news_index_path)

        # Build qrels of both relevant and non-relevant documents.
        qrels_dict = {}
        with open(qrels_path, 'r', encoding="utf-8") as qrels_file:
            # Read each line of qrels file.
            for line in qrels_file:
                if len(line) > 4:
                    query, _, doc_id, score = self.retrieval_utils.unpack_qrels_line(line)
                    if query not in qrels_dict:
                        qrels_dict[query] = [doc_id]
                    else:
                        qrels_dict[query].append(doc_id)
        query_counter = 0
        # Begin writing to run file.
        with open(run_path, "w", encoding='utf-8') as f_run:
            for query_id, valid_docs in qrels_dict.items():
                query_counter += 1
                print('query {} / {}'.format(query_counter, len(qrels_dict)))
                # Build query (limit to 'words' number of query terms).
                query_dict = json.loads(search_tools_news.get_contents_from_docid(query_id))
                query = self.process_query_news(query_dict=query_dict, query_type=query_type)
                query = " ".join(query.split(" ")[:words])

                # Get 'hits' many documents from entity index.
                try:
                    retrieved_hits = self.search(query=query, hits=hits)
                except:
                    retrieved_hits = self.search(query=query.encode('utf-8'), hits=hits)

                # Reduce retrieved docs to those within qrels.
                valid_hits = [i for i in retrieved_hits if i[0] in valid_docs]
                rank = 1
                # Write valid retrieved docs to file.
                for hit in valid_hits:
                    # Create and write run file.
                    run_line = " ".join((query_id, "Q0", hit[0], str(rank), "{:.6f}".format(hit[1]), "PYSERINI")) + '\n'
                    f_run.write(run_line)
                    # Next rank.
                    rank += 1

                # Add possible documents (i.e. within qrels) that are not retieved.
                if len(valid_hits) == 0:
                    missed_hits = valid_docs
                    min_score = 0.0
                else:
                    missed_hits = list(set(valid_docs) - set([hit[0] for hit in valid_hits]))
                    min_score = valid_hits[len(valid_hits)-1][1]
                # Write possible documents to file.
                for doc_id in missed_hits:
                    # Create and write run file.
                    min_score -= 0.1
                    run_line = " ".join((query_id, "Q0", doc_id, str(rank), "{:.6f}".format(min_score), "PYSERINI")) + '\n'
                    f_run.write(run_line)
                    # Next rank.
                    rank += 1


    def get_merge_qrels(self, qrels_entity_path_1, qrels_entity_path_2, qrels_passage_path_1, qrels_pasage_path_2,
                        fold_path, k=5):

        search_tools = SearchTools()
        entity_qrels_1 = search_tools.retrieval_utils.get_qrels_scaled_dict(qrels_entity_path_1)
        entity_qrels_2 = search_tools.retrieval_utils.get_qrels_scaled_dict(qrels_entity_path_2)
        passage_qrels_1 = search_tools.retrieval_utils.get_qrels_scaled_dict(qrels_passage_path_1)
        passage_qrels_2 = search_tools.retrieval_utils.get_qrels_scaled_dict(qrels_pasage_path_2)
        entity_merge_qrels = {**entity_qrels_1, **entity_qrels_2}
        passage_merge_qrels = {**passage_qrels_1, **passage_qrels_2}
        unique_keys = list(set(list(entity_merge_qrels.keys()) + list(passage_merge_qrels.keys())))

        min_size = int(len(unique_keys) / k)
        remainder = int(len(unique_keys) % k)
        remained_added = 0

        k_id_lists = []
        index_counter = 0
        for i in range(1, k+1):
            if remained_added < remainder:
                length = min_size + 1
                remained_added += 1
            else:
                length = min_size
            id_list = unique_keys[index_counter:index_counter+length]
            k_id_lists.append(id_list)
            index_counter += length

        for k_count, id_list in enumerate(k_id_lists):
            for query_type in ['passage', 'entity']:
                k_qrels_path = fold_path + str(k_count) + '_{}.qrels'.format(query_type)
                if query_type == 'passage':
                    merge_qrels = passage_merge_qrels
                else:
                    merge_qrels = entity_merge_qrels
                with open(k_qrels_path, 'w') as f_qrels:
                    for i in id_list:
                        if i in merge_qrels:
                            for doc_id, score in merge_qrels[i].items():
                                f_qrels.write(" ".join((i, '0', doc_id, str(score))) + '\n')

                k_topics_path = fold_path + str(k_count) + '_{}.topics'.format(query_type)
                with open(k_topics_path, 'w') as f_topics:
                    for i in id_list:
                        f_topics.write(i + '\n')


    def write_news_folds_to_runs(self, run_file_base, fold_topics_paths, run_paths, xml_topics_paths,
                                 ranking_type='passage', use_xml=False):
        """ """
        assert ranking_type == 'passage' or ranking_type == 'entity'

        if use_xml:
            id_map = {}
            for xml_topics_path in xml_topics_paths:
                passage_id_map, entity_id_map = self.get_news_ids_maps(xml_topics_path=xml_topics_path, ranking_type=ranking_type)
                if ranking_type == 'passage':
                    id_map.update(passage_id_map)
                else:
                    id_map.update(entity_id_map)
        else:
            id_map = {}

        run_data = []
        for run_path in run_paths:
            with open(run_path, 'r') as f_run:
                run_data += f_run.read().splitlines()
        print(run_data)

        print(id_map)
        # Expect folds to be 0 - K
        for fold_i, fold_path in enumerate(fold_topics_paths):
            fold_run_path = '{}_fold_{}.run'.format(run_file_base, fold_i)
            with open(fold_path, 'r') as f_fold:
                fold_ids = f_fold.read().splitlines()
            with open(fold_run_path, 'w') as f_run_fold:
                for line in run_data:
                    query, q, doc_id, rank, score, name = self.retrieval_utils.unpack_run_line(line=line)
                    if query in id_map:
                        query_id = id_map[query]
                    else:
                        query_id = query
                    if query_id in fold_ids:
                        f_run_fold.write(" ".join((query_id, q, doc_id, rank, score, name)) + '\n')

    def write_fold_data_folders(self, fold_dir_base, fold_topics_bases, k=5):
        """ """
        path_dicts = {}
        for k_i in range(k):
            passage_qrels_path = fold_topics_bases + '_{}_passage.qrels'.format(k_i)
            passage_topics_path = fold_topics_bases + '_{}_passage.topics'.format(k_i)
            entity_qrels_path = fold_topics_bases + '_{}_entity.qrels'.format(k_i)
            entity_topics_path = fold_topics_bases + '_{}_entity.topics'.format(k_i)
            path_dicts[k_i] = (passage_qrels_path, passage_topics_path, entity_qrels_path, entity_topics_path)

        for k_i in range(k):
            # make dir
            fold_dir_path = fold_dir_base + '_{}_data/'.format(k_i)
            os.mkdir(fold_dir_path)

            # Get ids
            remaining_ids = list(range(k))
            remaining_ids.remove(k_i)
            random.shuffle(remaining_ids)
            valid_i = remaining_ids.pop(0)

            # Test data
            for i, dataset in zip([k_i, valid_i], ['test', 'valid']):
                old_passage_test_qrels, old_passage_test_topics, old_entity_test_qrels, old_entity_test_topics = path_dicts[i]

                new_passage_test_qrels = fold_dir_path + 'passage_{}.qrels'.format(dataset)
                new_passage_test_topics = fold_dir_path + 'passage_{}.topics'.format(dataset)
                new_entity_test_qrels = fold_dir_path + 'entity_{}.qrels'.format(dataset)
                new_entity_test_topics = fold_dir_path + 'entity_{}.topics'.format(dataset)
                shutil.copy(old_passage_test_qrels, new_passage_test_qrels)
                shutil.copy(old_passage_test_topics, new_passage_test_topics)
                shutil.copy(old_entity_test_qrels, new_entity_test_qrels)
                shutil.copy(old_entity_test_topics, new_entity_test_topics)

            def update_file(new_path, old_path):
                with open(new_path, 'a+') as f_new:
                    with open(old_path, 'r') as f_old:
                        for line in f_old:
                            f_new.write(line.strip() + '\n')

            for train_i in remaining_ids:
                old_passage_train_qrels_i, old_passage_train_topics_i, old_entity_train_qrels_i, old_entity_train_topics_i = path_dicts[train_i]

                new_passage_train_qrels_i = fold_dir_path + 'passage_train.qrels'
                new_passage_train_topics_i = fold_dir_path + 'passage_train.topics'
                new_entity_train_qrels_i = fold_dir_path + 'entity_train.qrels'
                new_entity_train_topics_i = fold_dir_path + 'entity_train.topics'

                update_file(new_path=new_passage_train_qrels_i, old_path=old_passage_train_qrels_i)
                update_file(new_path=new_passage_train_topics_i, old_path=old_passage_train_topics_i)
                update_file(new_path=new_entity_train_qrels_i, old_path=old_entity_train_qrels_i)
                update_file(new_path=new_entity_train_topics_i, old_path=old_entity_train_topics_i)


###############################################################################
################################ Eval Class ###################################
###############################################################################


default_eval_config = [('map',None), ('Rprec',None), ('recip_rank', None), ('ndcg',20), ('P',20), ('recall',40),
                       ('recall',100), ('recall',1000)]

class EvalTools:

    """ Information retrieval eval class. """

    retrieval_utils = RetrievalUtils()

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
        self.query_metrics_run_sum = None
        self.query_metrics_oracle_sum = None
        self.qrels_dict = None
        self.metric_labels = None


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


    def get_query_metrics(self, run, R, eval_config):
        """ Build metrics (string and dict representation) from eval_config. """

        # Assert a valid eval metric config.
        assert isinstance(eval_config, list)
        for eval_func, k in eval_config:
            assert eval_func in self.implemented_metrics, "eval_func: {}, k: {}".format(eval_func, k)
            assert isinstance(k, int) or k == None, "eval_func: {}, k: {}".format(eval_func, k)
        # Loop over implemented_metrics/eval_config.
        query_metrics = ''
        query_metrics_dict = {}
        self.metric_labels = []
        for eval_func, k in eval_config:
            # Build label of metric.
            if k == None:
                metric_label = eval_func
            else:
                metric_label = eval_func + '_' + str(k)
            self.metric_labels.append(metric_label)
            # Calculate metric.
            metric = self.implemented_metrics[eval_func](run=run, k=k, R=R)
            # Append metric label and metric to string.
            query_metrics += metric_label + ' ' + "{:.6f}".format(metric) + ' '
            query_metrics_dict[metric_label] = metric

        return query_metrics, query_metrics_dict


    def __process_topic(self, query, run_doc_ids, eval_config):
        """ Process eval of topic. """
        # Assert query of run in qrels
        if query in self.qrels_dict:
            # For each doc_id find whether relevant (1) or not relevant (0), appending binary relevance to run.
            run = []
            for d in run_doc_ids:
                if d in self.qrels_dict[query]:
                    run.append(1.0)
                else:
                    run.append(0.0)

            # Calculate number of relevant docs in qrels (R).
            R = len(self.qrels_dict[query])
            # Build query metric string.
            _, query_metrics_run = self.get_query_metrics(run=run, R=R, eval_config=eval_config)
            self.query_metrics_run_sum = dict(Counter(self.query_metrics_run_sum) + Counter(query_metrics_run))

            # get oracle metrics
            run_oracle = sorted(run, reverse=True)
            _, query_metrics_oracle = self.get_query_metrics(run=run_oracle, R=R, eval_config=eval_config)
            self.query_metrics_oracle_sum = dict(Counter(self.query_metrics_oracle_sum) + Counter(query_metrics_oracle))
        else:
            print("query: {} not in qrels_dict".format(query))


    def write_eval_from_qrels_and_run(self, run_path, qrels_path, eval_path=None, eval_config=default_eval_config):
        """ Given qrels and run paths calculate evaluation metrics by query and aggreated and write to file. """
        self.query_metrics_run_sum = {}
        self.query_metrics_oracle_sum = {}
        self.qrels_dict = self.retrieval_utils.get_qrels_binary_dict(qrels_path=qrels_path)

        with open(run_path, 'r') as f_run:
            # Store query of run (i.e. previous query).
            topic_query = None
           # Store doc_ids of query got relevance mapping.
            run_doc_ids = []
            for line in f_run:
                # Assumes run file is written in ascending order i.e. rank=1, rank=2, etc.

                query, _, doc_id, _, _, _ = self.retrieval_utils.unpack_run_line(line=line)

                # If run batch complete.
                if (topic_query != None) and (topic_query != query):

                    self.__process_topic(query=topic_query, run_doc_ids=run_doc_ids, eval_config=eval_config)

                    # Start next query.
                    run_doc_ids = []

                # Add doc_id to run and map topic_query->query.
                topic_query = query
                run_doc_ids.append(doc_id)

        if len(run_doc_ids) > 0:

            self.__process_topic(query=topic_query, run_doc_ids=run_doc_ids, eval_config=eval_config)

        # Find mean of metrics.
        topic_count = len(self.qrels_dict)
        eval_metric = {k: v / topic_count for k, v in self.query_metrics_run_sum.items()}
        eval_metric_oracle = {k: v / topic_count for k, v in self.query_metrics_oracle_sum.items()}

        # Write overall eval to file.
        if eval_path == None:
            eval_path = run_path + '.eval'
        with open(eval_path, 'w') as f_eval:

            f_eval.write("Run:\n")
            for metric_label in self.metric_labels:
                f_eval.write(metric_label + '\t' + "{:.4f}".format(eval_metric[metric_label]) + '\n')

            f_eval.write("\nRe-ranking Oracle:\n")
            for metric_label in self.metric_labels:
                f_eval.write(metric_label + '\t' + "{:.4f}".format(eval_metric_oracle[metric_label]) + '\n')

        return eval_metric, eval_metric_oracle


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
                search_tools.write_run_car(topics_path=topics_path, run_path=run_path, hits=hits)

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
    # eval_config = [('map', None), ('Rprec',None), ('recip_rank', None), ('ndcg',5), ('P',20), ('recall',40)]
    # eval_tools = EvalTools()
    # run_path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/TREC-NEWS/2018/entity.custom_anserini.500000_doc.100_words.title.fixed_qrels.run'
    # qrels_path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/TREC-NEWS/2018/news_track.2018.entity.qrels'
    # eval_path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/TREC-NEWS/2018/entity.custom_anserini.500000_doc.100_words.title.fixed_qrels.run.summed.eval.map'
    # eval_tools.write_eval_from_qrels_and_run(run_path=run_path, qrels_path=qrels_path, eval_path=eval_path, eval_config=eval_config)

    # qrels_entity_path_1 = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/TREC-NEWS/5_fold/news_track.2018.entity.qrels'
    # qrels_entity_path_2 = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/TREC-NEWS/5_fold/news_track.2019.entity.qrels'
    # qrels_passage_path_1 = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/TREC-NEWS/5_fold/news_track.2018.passage.qrels'
    # qrels_pasage_path_2 = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/TREC-NEWS/5_fold/news_track.2019.passage.qrels'
    # fold_path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/TREC-NEWS/5_fold/scaled_5fold_'
    # search_tools = SearchTools()
    #
    # search_tools.get_merge_qrels(qrels_entity_path_1=qrels_entity_path_1,
    #                              qrels_entity_path_2=qrels_entity_path_2,
    #                              qrels_passage_path_1=qrels_passage_path_1,
    #                              qrels_pasage_path_2=qrels_pasage_path_2,
    #                              fold_path=fold_path)

    search_tools = SearchTools()
    run_file_base = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/TREC-NEWS/5_fold/anserini_bm25_rm3_default_passage_scaled'
    fold_topics_paths = ['/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/TREC-NEWS/5_fold/scaled_5fold_{}_entity.topics'.format(i) for i in [0,1,2,3,4]]
    run_paths = ['/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/TREC-NEWS/5_fold/background.2018.anserini.bm5.rm3.default.run',
                 '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/TREC-NEWS/5_fold/background.2019.anserini.bm5.rm3.default.run']
    xml_topics_paths = ['/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/TREC-NEWS/2018/newsir18-topics.txt',
                        '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/TREC-NEWS/2019/newsir19-background-linking-topics.xml']
    ranking_type = 'passage'
    use_xml = True
    search_tools.write_news_folds_to_runs(run_file_base=run_file_base,
                                          fold_topics_paths=fold_topics_paths,
                                          run_paths=run_paths,
                                          xml_topics_paths=xml_topics_paths,
                                          ranking_type=ranking_type,
                                          use_xml=use_xml)

    # search_tools = SearchTools()
    # fold_topics_bases = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/TREC-NEWS/5_fold/scaled_5fold'
    # fold_dir_base = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/TREC-NEWS/5_fold/scaled_5fold'
    # search_tools.write_fold_data_folders(fold_dir_base=fold_dir_base,
    #                                      fold_topics_bases=fold_topics_bases)

