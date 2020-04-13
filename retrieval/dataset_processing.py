
from retrieval.tools import SearchTools
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

import collections
import torch
import time
import six


class TrecCarProcessing:

    def __init__(self, qrels_path, run_path, index_path, data_dir_path,
                 tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'), max_length=512):

        self.qrels_path = qrels_path
        self.run_path = run_path
        self.index_path = index_path
        self.data_dir_path = data_dir_path
        self.search_tools = SearchTools(index_path=self.index_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qrels = self.get_qrels()
        self.input_ids_list = []
        self.token_type_ids_list = []
        self.attention_mask_list = []
        self.labels_list = []
        self.chuck_counter = 0
        self.chuck_query_size = 0


    def __convert_to_unicode(text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python 3?")


    def get_qrels(self):
        """Loads qrels into a dict of key: topic, value: list of relevant doc ids."""
        qrels = collections.defaultdict(list)
        with open(self.qrels_path) as f_qrels:
            for i, line in enumerate(f_qrels):
                topic_query, _, doc_id, relevance = line.rstrip().split(' ')
                if int(relevance) >= 1:
                    qrels[topic_query].append(doc_id)
                if i % 10000 == 0:
                    print('Loading qrels {}'.format(i))
        return qrels


    def __process_sequential_topic(self, topic_BERT_encodings):
        """ """
        for query, doc_id, BERT_encodings in topic_BERT_encodings:
            self.input_ids_list.append(BERT_encodings['input_ids'])
            self.token_type_ids_list.append(BERT_encodings['token_type_ids'])
            self.attention_mask_list.append(BERT_encodings['attention_mask'])
            if doc_id in self.qrels[query]:
                self.labels_list.append([1])
            else:
                self.labels_list.append([0])


    def __process_non_sequential_topic(self, topic_R_BERT_encodings, topic_N_BERT_encodings):
        """ """
        input_ids_list, token_type_ids_list, attention_mask_list, labels_list = [], [], [], []
        return input_ids_list, token_type_ids_list, attention_mask_list, labels_list


    def __write_chuck_to_directory(self):
        """ """
        input_ids_tensor = torch.tensor(self.input_ids_list)
        token_type_ids_tensor = torch.tensor(self.token_type_ids_list)
        attention_mask_tensor = torch.tensor(self.attention_mask_list)
        labels_tensor = torch.tensor(self.labels_list)

        dataset = TensorDataset(input_ids_tensor, token_type_ids_tensor, attention_mask_tensor, labels_tensor)

        path = os.path.join(self.data_dir_path, 'tensor_dataset_chuck_{}.pt'.format(self.chuck_counter))
        print('saving tensor to: {}'.format(path))
        torch.save(dataset, path)

        self.input_ids_list = []
        self.token_type_ids_list = []
        self.attention_mask_list = []
        self.labels_list = []


    def __process_topic(self, sequential, topic_BERT_encodings, topic_R_BERT_encodings, topic_N_BERT_encodings):
        """ """

        if sequential:
            self.__process_sequential_topic(topic_BERT_encodings=topic_BERT_encodings)
        else:
            self.__process_non_sequential_topic(topic_R_BERT_encodings=topic_R_BERT_encodings,
                                                topic_N_BERT_encodings=topic_N_BERT_encodings)

        self.chuck_counter += 1
        if self.chuck_counter % self.chuck_query_size == 0:
            print('WRITE DATA CHUCK')
            self.__write_chuck_to_directory()


    def build_dataset(self, sequential=False, chuck_query_size=1e8):
        """ """
        self.chuck_query_size = chuck_query_size
        #start_time = time.time()

        with open(self.run_path) as f_run:
            # Stores data for sequential dataset
            topic_BERT_encodings = []
            # Stores data for non-sequential dataset.
            # Having data in relevant (R) and not relevant (N) allows to added extra R samples.
            topic_R_BERT_encodings = []
            topic_N_BERT_encodings = []
            # Store previous query so we know when a new topic began.
            topic_query = None
            for i, line in enumerate(f_run):
                # Unpack line in run file.
                query, _, doc_id, rank, _, _ = line.split(' ')

                # If final doc_id in topic -> process batch.
                if (topic_query != None) and (topic_query != query):
                    self.__process_topic(sequential=sequential,
                                         topic_BERT_encodings=topic_BERT_encodings,
                                         topic_R_BERT_encodings=topic_R_BERT_encodings,
                                         topic_N_BERT_encodings=topic_N_BERT_encodings)

                # Decode query.
                decoded_query = self.search_tools.decode_query(q=query)
                # Extract text from index using doc_id.
                text = self.search_tools.get_contents_from_docid(doc_id=doc_id)
                # Get BERT inputs {input_ids, token_type_ids, attention_mask} -> [CLS] Q [SEP] DOC [SEP]
                BERT_encodings = self.tokenizer.encode_plus(text=decoded_query, text_pair=text, max_length=max_length,
                                                            add_special_tokens=True, pad_to_max_length=True)

                # Append doc_id data topic
                if sequential == True:
                    topic_BERT_encodings.append((query, doc_id, BERT_encodings))
                else:
                    if doc_id in self.qrels[topic_query]:
                        topic_R_BERT_encodings.append(BERT_encodings)
                    else:
                        topic_N_BERT_encodings.append(BERT_encodings)

                # Store query as topic query.
                topic_query = query

        # TODO
        self.__process_topic(sequential=sequential,
                             topic_BERT_encodings=topic_BERT_encodings,
                             topic_R_BERT_encodings=topic_R_BERT_encodings,
                             topic_N_BERT_encodings=topic_N_BERT_encodings)


if __name__ == '__main__':
    import os

    # index_path = '/Users/iain/LocalStorage/anserini_index/car_entity_v9'
    # run_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'test.pages.cbor-hierarchical.entity.run.decode')
    # qrels_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'test.pages.cbor-hierarchical.entity.qrels')
    # data_dir_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'results')

    index_path = '/nfs/trec_car/index/anserini_paragraphs/lucene-index.car17v2.0.paragraphsv2'
    run_path = '/nfs/trec_car/data/bert_reranker_datasets/test_100.run'
    qrels_path = '/nfs/trec_car/data/bert_reranker_datasets/test_100.qrels'
    data_dir_path = '/nfs/trec_car/data/bert_reranker_datasets/'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 512
    processing = TrecCarProcessing(qrels_path=qrels_path,
                                   run_path=run_path,
                                   index_path=index_path,
                                   data_dir_path=data_dir_path,
                                   tokenizer=tokenizer,
                                   max_length=max_length)
    processing.build_dataset()