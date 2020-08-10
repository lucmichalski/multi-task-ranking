
from protocol_buffers import document_pb2
from retrieval.tools import SearchTools

from REL.mention_detection import MentionDetection
from REL.utils import process_results
from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import Cmns, load_flair_ner

import pandas as pd
import urllib
import pickle
import stream
import json
import time
import lmdb
import re
import os


class TrecNewsParser:
    """ Class to parse TREC News (Washington Post) to protocol buffer Document  (protocol_buffers/document_pb2/Document)
    """

    def __init__(self, rel_wiki_year, rel_base_url, rel_model_path, car_id_to_name_path):

        self.document = None
        self.mention_detection = None
        self.tagger_ner = None
        self.entity_disambiguation = None
        self.car_id_to_name_path = None
        self.not_valid_counter = 0
        self.valid_counter = 0
        self.__init_rel_models(rel_wiki_year, rel_base_url, rel_model_path, car_id_to_name_path)


    def write_documents_to_file(self, path, documents, buffer_size=10):
        """ Write list of Documents messages to binary file. """
        stream.dump(path, *documents, buffer_size=buffer_size)


    def get_list_protobuf_messages(self, path):
        """ Retrieve list of protocol buffer messages from binary fire """
        return [d for d in stream.parse(path, document_pb2.Document)]


    def get_protobuf_message(self, path, doc_id):
        """ Retrieve protocol buffer message matching 'doc_id' from binary fire """
        return [d for d in stream.parse(path, document_pb2.Document) if d.doc_id == doc_id][0]


    def __get_entity_link_totals_from_document_contents(self, document_contents, link_type='REL'):
        """ Append entity_link_totals features (protocol_buffers/document.proto:EntityLinkTotals) from
        document_contents. """
        assert link_type == "REL", "link_type: {} not 'REL' flag".format(link_type)

        def get_correct_entity_links(document_content, link_type):
            """ Return list of EntityLink message for either 'MANUAL' or 'SYNTHETIC' links. """
            if link_type == 'REL':
                return document_content.rel_entity_links

        # Build set of unique 'entity_id's of entities linked in document_contents
        entity_ids = []
        for document_content in document_contents:
            entity_links = get_correct_entity_links(document_content=document_content, link_type=link_type)

            for entity_link in entity_links:
                entity_ids.append(entity_link.entity_id)
        unique_entity_ids = set(entity_ids)

        entity_link_totals = []
        for unique_entity_id in unique_entity_ids:
            # For each unique 'entity_id' loop over 'document_contents' to build anchor_text_data.
            # 'anchor_text_data': key= 'anchor_text' of 'entity_id', value= number of times 'anchor_text' linked to
            # 'entity_id' i.. frequency.
            anchor_text_data = {}
            for document_content in document_contents:
                entity_links = get_correct_entity_links(document_content=document_content, link_type=link_type)

                for entity_link in entity_links:
                    if entity_link.entity_id == unique_entity_id:
                        entity_name = entity_link.entity_name
                        anchor_text = entity_link.anchor_text

                        if anchor_text not in anchor_text_data:
                            anchor_text_data[anchor_text] = 1
                        else:
                            anchor_text_data[anchor_text] += 1

            # For each unique 'entity_id' transform 'anchor_text_data' to required data format for 'anchor_text_frequency'
            anchor_text_frequencies = []
            for k, v in anchor_text_data.items():
                anchor_text_frequency = document_pb2.EntityLinkTotal.AnchorTextFrequency()
                anchor_text_frequency.anchor_text = k
                anchor_text_frequency.frequency = v
                anchor_text_frequencies.append(anchor_text_frequency)

            # Create an EntityLinkTotal message for 'entity_id'.
            entity_link_total = document_pb2.EntityLinkTotal()
            entity_link_total.entity_id = unique_entity_id
            entity_link_total.entity_name = entity_name
            entity_link_total.anchor_text_frequencies.extend(anchor_text_frequencies)

            # Append to Document.entity_link_totals
            entity_link_totals.append(entity_link_total)

        return entity_link_totals


    def __add_entity_link_totals(self):
        """ """
        # Add list of synthetically tagged EntityLinkTotal messages by parsing list of DocumentContents messages.
        rel_entity_link_totals = self.__get_entity_link_totals_from_document_contents(
            document_contents=self.document.document_contents, link_type='REL')
        self.document.rel_entity_link_totals.extend(rel_entity_link_totals)


    def parse_article_to_protobuf(self, article):
        """ """

        # Initialise empty message.
        self.document = document_pb2.Document()
        self.document.doc_id = article['id']
        self.document.doc_name = article['title']

        document_content = document_pb2.DocumentContent()
        document_content.content_id = article['id']
        document_content.content_type = 1
        document_content.text = article['text']

        self.document.document_contents.append(document_content)

        self.__add_rel_entity_links()

        self.__add_entity_link_totals()

        return self.document


    def __build_text_from_contents(self, title, contents):
        """ """
        # Extract conents text
        content_text = ""
        for content in contents:
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
            return title + '. ' + content_text
        else:
            return content_text


    def build_article_from_json(self, article_json):
        """ """

        id = article_json['id']

        if isinstance(article_json['title'], str):
            title = article_json['title']
        else:
            title = ""

        text = self.__build_text_from_contents(title=title, contents=article_json['contents'])

        return {
            'id': id,
            'title': title,
            'text': text
        }


    def __rel_id_to_car_id(self, rel_id):
        """ Build TREC CAR id from REL id (will be close, roughly 90% converage)."""
        return 'enwiki:' + urllib.parse.quote(rel_id.replace('_', ' '), encoding='utf-8')


    def __add_rel_entity_links(self):
        """ Add REL entity linker links to all document contents. """

        # Build batch REL text input - sharding document contents into sentences.
        processed_document_contents = {}
        for document_content in self.document.document_contents:
            content_id = document_content.content_id
            text = document_content.text
            for i, sentence in enumerate(text.split(".")):
                sentence_id = str(content_id + (str(i)))
                processed_document_contents[sentence_id] = [str(sentence), []]

        # Run REL model with document contents.
        mentions_dataset, n_mentions = self.mention_detection.find_mentions(processed_document_contents,
                                                                            self.tagger_ner)
        predictions, timing = self.entity_disambiguation.predict(mentions_dataset)
        entity_links_dict = process_results(mentions_dataset, predictions, processed_document_contents)

        # Connet to LMDB of: {pickle(car_id): pickle(car_name).}
        env = lmdb.open(self.car_id_to_name_path, map_size=2e10)
        with env.begin(write=False) as txn:

            for document_content in self.document.document_contents:

                content_id = document_content.content_id
                text = document_content.text

                i_sentence_start = 0
                for i, sentence in enumerate(text.split(".")):
                    sentence_id = str(content_id + (str(i)))
                    if sentence_id in entity_links_dict:
                        for entity_link in entity_links_dict[sentence_id]:
                            # % of confidence in entity linking
                            if float(entity_link[4]) >= 0.0:
                                i_start = i_sentence_start + entity_link[0] + 1
                                i_end = i_start + entity_link[1]
                                span_text = text[i_start:i_end]

                                entity_id = self.__rel_id_to_car_id(rel_id=entity_link[3])
                                entity_name_pickle = txn.get(pickle.dumps(entity_id))

                                if entity_name_pickle != None:
                                    self.valid_counter += 1
                                    entity_name = pickle.loads(entity_name_pickle)
                                    #entity_name = entity_id

                                    if entity_link[2] == span_text:

                                        assert entity_link[2] == span_text, \
                                            "word_text: '{}' , text[start_i:end_i]: '{}'".format(entity_link[2], span_text)

                                        anchor_text_location = document_pb2.EntityLink.AnchorTextLocation()
                                        anchor_text_location.start = i_start
                                        anchor_text_location.end = i_end

                                        # Create new EntityLink message.
                                        rel_entity_link = document_pb2.EntityLink()
                                        rel_entity_link.anchor_text = entity_link[2]
                                        rel_entity_link.entity_id = entity_id
                                        rel_entity_link.entity_name = entity_name
                                        rel_entity_link.anchor_text_location.MergeFrom(anchor_text_location)

                                        document_content.rel_entity_links.append(rel_entity_link)

                                    else:
                                        regex = re.escape(entity_link[2])
                                        for match in re.finditer(r'{}'.format(regex), sentence):
                                            i_start = i_sentence_start + match.start()
                                            i_end = i_sentence_start + match.end()
                                            span_text = text[i_start:i_end]

                                            assert entity_link[2] == span_text, \
                                                "word_text: '{}' , text[start_i:end_i]: '{}'".format(entity_link[2], span_text)

                                            anchor_text_location = document_pb2.EntityLink.AnchorTextLocation()
                                            anchor_text_location.start = i_start
                                            anchor_text_location.end = i_end

                                            # Create new EntityLink message.
                                            rel_entity_link = document_pb2.EntityLink()
                                            rel_entity_link.anchor_text = entity_link[2]
                                            rel_entity_link.entity_id = entity_id
                                            rel_entity_link.entity_name = entity_name
                                            rel_entity_link.anchor_text_location.MergeFrom(anchor_text_location)

                                            document_content.rel_entity_links.append(rel_entity_link)

                    i_sentence_start += len(sentence) + 1


    def __init_rel_models(self, rel_wiki_year, rel_base_url, rel_model_path, car_id_to_name_path):
        """ """
        # Will require models and data saved paths
        wiki_version = "wiki_" + rel_wiki_year
        self.mention_detection = MentionDetection(rel_base_url, wiki_version)
        self.tagger_ner = load_flair_ner("ner-fast")
        config = {
            "mode": "eval",
            "model_path": rel_model_path,
        }
        self.entity_disambiguation = EntityDisambiguation(rel_base_url, wiki_version, config)
        self.car_id_to_name_path = car_id_to_name_path


    def parse_run_file_to_parquet(self, run_path,  index_path, chunks=1, write_output=False, dir_path=None,
                                  num_docs=10, print_intervals=1):
        """ """

        # create new dir to store data chunks
        if (os.path.isdir(dir_path) == False) and write_output:
            print('making dir: {}'.format(dir_path))
            os.mkdir(dir_path)

        def write_chunk(article_data, dir_path, chunk):
            """ write data chunks to parquet """
            parquet_path = dir_path + 'article_data_chunk_' + str(chunk) + '.parquet'
            columns = ['query', 'doc_id', 'rank', 'article', 'article_bytearray']
            pd.DataFrame(article_data, columns=columns).to_parquet(parquet_path)

        t_start = time.time()

        search_tools = SearchTools(index_path=index_path)

        chunk = 0
        article_data = []
        with open(run_path, "r") as f_run:
            # Loop over topics.
            i = 0
            for line in f_run:

                query, _, doc_id, rank, _, _ = search_tools.retrieval_utils.unpack_run_line(line)

                article_json = self.build_article_from_json(json.loads(search_tools.get_contents_from_docid(doc_id=doc_id)))
                doc = self.parse_article_to_protobuf(article=article_json)
                article_data.append([query, doc_id, rank, article_json, bytearray(pickle.dumps(doc.SerializeToString()))])

                if i + 1 > num_docs:
                    break

                # write data chunk to file
                if ((i + 1) % chunks == 0) and (i != 0 or num_docs == 1):
                    if write_output:
                        print('WRITING TO FILE: {}'.format(i))
                        write_chunk(article_data=article_data, dir_path=dir_path, chunk=chunk)

                        # begin new list
                        article_data = []
                        chunk += 1

                if ((i + 1) % print_intervals == 0):
                    print('----- DOC #{} -----'.format(i))
                    print(self.document.doc_id)
                    time_delta = time.time() - t_start
                    print('time elapse: {} --> time / page: {}'.format(time_delta, time_delta / (i + 1)))

                i += 1

        if write_output and (len(article_data) > 0):
            print('WRITING FINAL FILE: {}'.format(i))
            write_chunk(article_data=article_data, dir_path=dir_path, chunk=chunk)


    def parse_json_to_protobuf(self, read_path, num_docs, write_output=False, write_path=None, print_intervals=1000,
                               buffer_size=1000):
        """ """
        t_start = time.time()

        documents = []
        with open(read_path, encoding='utf-8') as txt_file:
            for i, line in enumerate(txt_file):
                article_json = json.loads(line)

                article = self.build_article_from_json(article_json)

                if i+1 > num_docs:
                    break

                # parse page to create new document.
                # Append Document message to document list.
                documents.append(self.parse_article_to_protobuf(article=article))

                if ((i + 1) % print_intervals == 0):
                    print('----- DOC #{} -----'.format(i))
                    print(self.document.doc_id)
                    time_delta = time.time() - t_start
                    print('time elapse: {} --> time / page: {}'.format(time_delta, time_delta / (i + 1)))

        if write_output:
            print('STEAMING DATA TO FILE: {}'.format(write_path))
            self.write_documents_to_file(path=write_path, documents=documents, buffer_size=buffer_size)
            print('FILE WRITTEN')

        time_delta = time.time() - t_start
        print('PROCESSED DATA: {} --> processing time / page: {}'.format(time_delta, time_delta / (i + 1)))


if __name__ == '__main__':

    path = '/Users/iain/LocalStorage/TREC-NEWS/WashingtonPost.v2/data/TREC_Washington_Post_collection.v2.jl'
    rel_base_url = "/Users/iain/LocalStorage/coding/github/REL/"
    rel_wiki_year = '2019'
    rel_model_path = "/Users/iain/LocalStorage/coding/github/REL/ed-wiki-{}/model".format(rel_wiki_year)
    car_id_to_name_path = '/Users/iain/LocalStorage/lmdb.map_id_to_name.v1'
    print_intervals = 100
    num_docs = 500
    chunks = 100
    write_output = True
    dir_path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/2018_bm25_chunks_full_v3/'
    tnp = TrecNewsParser(rel_wiki_year=rel_wiki_year,
                         rel_base_url=rel_base_url,
                         rel_model_path=rel_model_path,
                         car_id_to_name_path=car_id_to_name_path)

    index_path = '/Users/iain/LocalStorage/TREC-NEWS/lucene-index-copy'
    run_path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/anserini.bm5.default.run'
    tnp.parse_run_file_to_parquet(run_path=run_path,
                                  index_path=index_path,
                                  write_output=write_output,
                                  dir_path=dir_path,
                                  num_docs=num_docs,
                                  chunks=chunks,
                                  print_intervals=print_intervals)
