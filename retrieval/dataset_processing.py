
from retrieval.tools import SearchTools
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

import collections
import torch
import time
import six
import os


class TrecCarProcessing:
    """ Process TREC CAR qrels and run files to Pytorch datasets. """

    def __init__(self, qrels_path, run_path, index_path, data_dir_path,
                 tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'), max_length=512):
        # Path to qrels file.
        self.qrels_path = qrels_path
        # Path to run file.
        self.run_path = run_path
        # Path to Anserini/Lucene index for accessing text.
        self.index_path = index_path
        # Path to data directory to write output PyTorch file(s).
        self.data_dir_path = data_dir_path
        # Initialise searching capabilities over Anserini/Lucene index.
        self.search_tools = SearchTools(index_path=self.index_path)
        # Tokenizer function (text -> BERT tokens)
        self.tokenizer = tokenizer
        # Max length of BERT tokens.
        self.max_length = max_length
        # load qrels dictionary {query: [doc_id, doc_id, etc.]} into memory.
        self.qrels = self.get_qrels()
        # Lists of BERT inputs
        self.input_ids_list = []
        self.token_type_ids_list = []
        self.attention_mask_list = []
        self.labels_list = []
        # Counter of current chuck being processed.
        self.chuck_counter = None
        # Count number of topics being processed.
        self.topic_counter = None
        # Number of topics processed in each chuck before being processed.
        self.chuck_topic_size = None


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
                query, _, doc_id, _ = line.rstrip().split(' ')
                qrels[query].append(doc_id)
                if i % 1000 == 0:
                    print('Loaded #{} lines in qrels file'.format(i))
        print("Loaded qrels files (#{} lines)".format(i))
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


    def __process_topic(self, sequential, topic_BERT_encodings, topic_R_BERT_encodings, topic_N_BERT_encodings):
        """ """
        if sequential:
            self.__process_sequential_topic(topic_BERT_encodings=topic_BERT_encodings)
        else:
            self.__process_non_sequential_topic(topic_R_BERT_encodings=topic_R_BERT_encodings,
                                                topic_N_BERT_encodings=topic_N_BERT_encodings)


    def __write_chuck_to_directory(self):
        """ """
        print('Building chuck #{}'.format(self.chuck_counter))

        input_ids_tensor = torch.tensor(self.input_ids_list)
        token_type_ids_tensor = torch.tensor(self.token_type_ids_list)
        attention_mask_tensor = torch.tensor(self.attention_mask_list)
        labels_tensor = torch.tensor(self.labels_list)
        print('input_ids_tensor shape: {}'.format(input_ids_tensor.shape))
        print('token_type_ids_tensor shape: {}'.format(token_type_ids_tensor.shape))
        print('attention_mask_tensor shape: {}'.format(attention_mask_tensor.shape))
        print('labels_tensor shape: {}'.format(labels_tensor.shape))

        dataset = TensorDataset(input_ids_tensor, token_type_ids_tensor, attention_mask_tensor, labels_tensor)

        path = os.path.join(self.data_dir_path, 'tensor_dataset_chuck_{}.pt'.format(self.chuck_counter))
        print('saving tensor to: {}'.format(path))
        torch.save(obj=dataset, f=path)

        self.input_ids_list = []
        self.token_type_ids_list = []
        self.attention_mask_list = []
        self.labels_list = []

        self.chuck_counter += 1


    def build_dataset(self, sequential=False, chuck_topic_size=1e8):
        """ """
        # Counter of current chuck being processed.
        self.chuck_counter = 0
        # Count number of topics being processed.
        self.topic_counter = 0
        # Number of topics processed in each chuck before being processed.
        self.chuck_topic_size = chuck_topic_size

        with open(self.run_path) as f_run:
            # Stores data for sequential dataset
            topic_BERT_encodings = []
            # Stores data for non-sequential dataset.
            # Having data in relevant (R) and not relevant (N) allows to added extra R samples.
            topic_R_BERT_encodings = []
            topic_N_BERT_encodings = []
            # Store previous query so we know when a new topic began.
            topic_query = None
            for line in f_run:
                # Unpack line in run file.
                query, _, doc_id, rank, _, _ = line.split(' ')

                # If final doc_id in topic -> process batch.
                if (topic_query != None) and (topic_query != query):
                    print(topic_query)

                    self.__process_topic(sequential=sequential,
                                         topic_BERT_encodings=topic_BERT_encodings,
                                         topic_R_BERT_encodings=topic_R_BERT_encodings,
                                         topic_N_BERT_encodings=topic_N_BERT_encodings)
                    self.topic_counter += 1
                    if self.topic_counter % self.chuck_topic_size == 0:
                        print('WRITE DATA CHUCK')
                        self.__write_chuck_to_directory()

                # Decode query.
                decoded_query = self.search_tools.decode_query(q=query)
                # Extract text from index using doc_id.
                text = self.search_tools.get_contents_from_docid(doc_id=doc_id)
                # Get BERT inputs {input_ids, token_type_ids, attention_mask} -> [CLS] Q [SEP] DOC [SEP]
                BERT_encodings = self.tokenizer.encode_plus(text=decoded_query,
                                                            text_pair=text,
                                                            max_length=self.max_length,
                                                            add_special_tokens=True,
                                                            pad_to_max_length=True)
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

        self.__write_chuck_to_directory()


if __name__ == '__main__':
    import os

    index_path = '/Users/iain/LocalStorage/anserini_index/car_entity_v9'
    run_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'test.pages.cbor-hierarchical.entity.small.run')
    qrels_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'test.pages.cbor-hierarchical.entity.small.qrels')
    data_dir_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'results')

    # index_path = '/nfs/trec_car/index/anserini_paragraphs/lucene-index.car17v2.0.paragraphsv2'
    # run_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmark_Y1_25.run'
    # qrels_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmark_Y1_25.qrels'
    # data_dir_path = '/nfs/trec_car/data/bert_reranker_datasets/'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 512
    processing = TrecCarProcessing(qrels_path=qrels_path,
                                   run_path=run_path,
                                   index_path=index_path,
                                   data_dir_path=data_dir_path,
                                   tokenizer=tokenizer,
                                   max_length=max_length)
    processing.build_dataset(sequential=True)