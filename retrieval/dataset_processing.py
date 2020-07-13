
from retrieval.tools import SearchTools, EvalTools, RetrievalUtils
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from metadata import NewsPassagePaths

import collections
import random
import torch
import time
import json
import six
import os


class DatasetProcessing:
    """ Process TREC CAR qrels and run files to Pytorch datasets. """

    retrieval_utils = RetrievalUtils()

    def __init__(self, qrels_path, run_path, index_path, data_dir_path, max_length=512, context_path=None,
                tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')):

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
        # Max length of BERT tokens.
        self.max_length = max_length
        # Path to context file containing text + context for most relevant entities
        self.context_path = context_path
        if self.context_path != None:
            with open(context_path) as json_file:
                # Load json from 'context_path' to dict
                self.context_dict = json.load(json_file)
        else:
            self.context_dict = {}
        # Tokenizer function (text -> BERT tokens)
        self.tokenizer = tokenizer
        # load qrels dictionary {query: [doc_id, doc_id, etc.]} into memory.
        self.qrels = self.retrieval_utils.get_qrels_dict(qrels_path=self.qrels_path)
        # Lists of BERT inputs.
        self.input_ids_list = []
        self.token_type_ids_list = []
        self.attention_mask_list = []
        self.labels_list = []
        # Lists used to build BERT inputs.
        # Stores data for sequential dataset
        self.topic_BERT_encodings = []
        # Having data in relevant (R) and not relevant (N) allows to added extra R samples.
        self.topic_R_BERT_encodings = []
        self.topic_N_BERT_encodings = []
        # Counter of current chuck being processed.
        self.chuck_counter = None
        # Count number of topics being processed.
        self.topic_counter = None
        # Number of topics processed in each chuck before being processed.
        self.chuck_topic_size = None


    def __process_sequential_topic(self):
        """ Process sequentially and do not even classes through sampling. Used for building validation dataset. """
        for query, doc_id, BERT_encodings in self.topic_BERT_encodings:
            # Append BERT model inputs.
            self.input_ids_list.append(BERT_encodings['input_ids'])
            self.token_type_ids_list.append(BERT_encodings['token_type_ids'])
            self.attention_mask_list.append(BERT_encodings['attention_mask'])
            # Qrels, query and doc_id used to determine whether entry in relevant or not relevant.
            if doc_id in self.qrels[query]:
                self.labels_list.append([1])
            else:
                self.labels_list.append([0])

        # New topics.
        self.topic_BERT_encodings = []
        self.topic_R_BERT_encodings = []
        self.topic_N_BERT_encodings = []


    def __process_non_sequential_topic_even_classes(self):
        """ Process sequentially and even classes through sampling. Used for building training dataset. """
        # Calculate number of relevant (R_count) and non relevant (N_count) documents in topic run.
        R_count = len(self.topic_R_BERT_encodings)
        N_count = len(self.topic_N_BERT_encodings)

        # If cannot balance classes (i.e. 0 relevant or 0 non relevant) do not add to dataset.
        if (R_count == 0) or (N_count == 0):
            self.topic_BERT_encodings = []
            self.__process_sequential_topic()
            return

        def add_extra_sample(BERT_encodings, diff):
            """ Even classes by sampling extra from . """
            idx_list = list(range(len(BERT_encodings)))
            # randomly sample diff number of samples.
            for idx in random.choices(idx_list, k=diff):
                BERT_encodings.append(BERT_encodings[idx])

        # If less relevant documents sample extra relevant documents.
        if R_count < N_count:
            diff = abs(R_count - N_count)
            add_extra_sample(BERT_encodings=self.topic_R_BERT_encodings, diff=diff)
        # If less non-relevant documents sample extra non-relevant documents.
        elif R_count > N_count:
            diff = abs(R_count - N_count)
            add_extra_sample(BERT_encodings=self.topic_N_BERT_encodings, diff=diff)

        # Assert length of encodings are equal.
        assert len(self.topic_N_BERT_encodings) == len(self.topic_R_BERT_encodings)
        # Add topics and shuffle data.
        self.topic_BERT_encodings = self.topic_R_BERT_encodings + self.topic_N_BERT_encodings
        random.shuffle(self.topic_BERT_encodings)

        # Process sequentially topics.
        self.__process_sequential_topic()


    def __process_topic(self, training_dataset):
        """ Process topic - whether sequential (validation) or not sequential (training). """
        if training_dataset == False:
            # Sequential (test/validation dataset).
            self.__process_sequential_topic()
        else:
            # Not sequential (training dataset).
            self.__process_non_sequential_topic_even_classes()

        # Update topic counter.
        self.topic_counter += 1


    def __write_chuck_to_directory(self):
        """ Write data chuck to Pytorch TensorDataset and initialise new data chuck."""
        # Create data_dir_path if does not exist.
        if os.path.isdir(self.data_dir_path) == False:
            print('Making directory: {}'.format(self.data_dir_path))
            os.mkdir(self.data_dir_path)

        print('Building chuck #{}'.format(self.chuck_counter))

        # Make tensor dataset from 4x tensors (input_ids, token_type_ids, attention_mask and labels).
        dataset = TensorDataset(torch.tensor(self.input_ids_list),
                                torch.tensor(self.token_type_ids_list),
                                torch.tensor(self.attention_mask_list),
                                torch.tensor(self.labels_list))


        # Save tensor dataset to data_dir_path.
        path = os.path.join(self.data_dir_path, 'tensor_dataset_chuck_{}.pt'.format(self.chuck_counter))
        print('saving tensor to: {}'.format(path))
        torch.save(obj=dataset, f=path)

        # Initialise new data chuck.
        self.chuck_counter += 1
        # New lists of BERT inputs.
        self.input_ids_list = []
        self.token_type_ids_list = []
        self.attention_mask_list = []
        self.labels_list = []


    def __get_encodings(self, text, max_length):
        """ Get encodings for context"""
        return self.tokenizer.encode_plus(text=text,
                                          max_length=max_length,
                                          add_special_tokens=False,
                                          pad_to_max_length=True)


    def build_car_dataset(self, training_dataset=False, chuck_topic_size=1e8, first_para=False):
        """ Build dataset and save data chucks of data_dir_path. If sequential flag is True (validation dataset) and if
        False (training dataset). """
        if training_dataset:
            print("** Building training dataset **")
        else:
            print("** Building test/validation dataset **")
        # Counter of current chuck being processed.
        self.chuck_counter = 0
        # Count number of topics being processed.
        self.topic_counter = 0
        # Number of topics processed in each chuck before being processed.
        self.chuck_topic_size = chuck_topic_size

        with open(self.run_path) as f_run:

            # Store previous query so we know when a new topic began.
            topic_query = None
            for line in f_run:
                # Unpack line in run file.
                query, _, doc_id, rank, _, _ = line.split(' ')
                # Assert correct query format..
                assert self.retrieval_utils.test_valid_line(line=line), "Not valid query: {}".format(line)

                # If final doc_id in topic -> process batch.
                if (topic_query != None) and (topic_query != query):

                    # Process topic
                    self.__process_topic(training_dataset=training_dataset)

                    # If specified data chuck size -> write chuck to file.
                    if self.topic_counter % self.chuck_topic_size == 0:
                        # write final chuck to file.
                        self.__write_chuck_to_directory()

                # Decode query.
                decoded_query = self.search_tools.decode_query(q=query)
                # Extract text from index using doc_id.
                if self.context_path == None:
                    text = self.search_tools.get_contents_from_docid(doc_id=doc_id)
                    if first_para:
                        text = text.split('\n')[0]

                    # Get BERT inputs {input_ids, token_type_ids, attention_mask} -> [CLS] Q [SEP] DOC [SEP]
                    BERT_encodings = self.tokenizer.encode_plus(text=decoded_query,
                                                                text_pair=text,
                                                                max_length=self.max_length,
                                                                add_special_tokens=True,
                                                                pad_to_max_length=True)
                else:
                    try:

                        text = self.context_dict[doc_id]['first_para'] + self.context_dict[doc_id]['top_ents']

                    except:
                        print('Failed to add context to: {}'.format(doc_id))
                        text = self.search_tools.get_contents_from_docid(doc_id=doc_id)
                        if first_para:
                            text = text.split('\n')[0]

                    # Add text and entity encodings
                    BERT_encodings = self.tokenizer.encode_plus(text=decoded_query,
                                                                text_pair=text,
                                                                max_length=self.max_length,
                                                                add_special_tokens=True,
                                                                pad_to_max_length=True)

                data = (query, doc_id, BERT_encodings)
                # Append doc_id data topic
                if training_dataset:
                    if doc_id in self.qrels[query]:
                        self.topic_R_BERT_encodings.append(data)
                    else:
                        self.topic_N_BERT_encodings.append(data)
                else:
                    self.topic_BERT_encodings.append(data)

                # Store query as topic query.
                topic_query = query

        # Process any queries remaining.
        self.__process_topic(training_dataset=training_dataset)

        # write final chuck to file.
        self.__write_chuck_to_directory()


    def build_news_dataset(self, training_dataset=False, chuck_topic_size=1e8, ranking_type='passage',
                           query_type='title+contents', car_index_path=None, xml_topics_path=None):
        """ Build dataset and save data chucks of data_dir_path. If sequential flag is True (validation dataset) and if
        False (training dataset). """

        if training_dataset:
            print("** Building training dataset **")
        else:
            print("** Building test/validation dataset **")
        # Counter of current chuck being processed.
        self.chuck_counter = 0
        # Count number of topics being processed.
        self.topic_counter = 0
        # Number of topics processed in each chuck before being processed.
        self.chuck_topic_size = chuck_topic_size

        if ranking_type == 'passage':
            passage_id_map, entity_id_map = self.search_tools.get_news_ids_maps(xml_topics_path=xml_topics_path,
                                                                                ranking_type=ranking_type)
        else:
            search_tools_car = SearchTools(index_path=car_index_path)

        with open(self.run_path) as f_run:

            # Store previous query so we know when a new topic began.
            topic_query = None
            for line in f_run:
                # Unpack line in run file.
                query_id, _, doc_id, rank, _, _ = self.retrieval_utils.unpack_run_line(line=line)

                # If final doc_id in topic -> process batch.
                if (topic_query != None) and (topic_query != query):

                    # Process topic
                    self.__process_topic(training_dataset=training_dataset)

                    # If specified data chuck size -> write chuck to file.
                    if self.topic_counter % self.chuck_topic_size == 0:
                        # write final chuck to file.
                        self.__write_chuck_to_directory()

                if ranking_type == 'passage':
                    # Decode query.
                    query_id = passage_id_map[query_id]
                    query_dict = json.loads(self.search_tools.get_contents_from_docid(doc_id=query_id))
                    query = self.search_tools.process_news_query(query_dict=query_dict, query_type=query_type)
                else:
                    query_dict = self.search_tools.get_contents_from_docid(doc_id=query_id)
                    query = self.search_tools.process_news_query(query_dict=query_dict, query_type=query_type)

                # Extract text from index using doc_id.
                if self.context_path == None:
                    if ranking_type == 'passage':
                        text_dict = json.loads(self.search_tools.get_contents_from_docid(doc_id=doc_id)
                        text = self.search_tools.process_news_query(query_dict=text_dict, query_type=query_type)
                    else:
                        text = search_tools_car.get_contents_from_docid(doc_id=doc_id)

                    # Get BERT inputs {input_ids, token_type_ids, attention_mask} -> [CLS] Q [SEP] DOC [SEP]
                    BERT_encodings = self.tokenizer.encode_plus(text=query,
                                                                text_pair=text,
                                                                max_length=self.max_length,
                                                                add_special_tokens=True,
                                                                pad_to_max_length=True,
                                                                truncation_strategy='longest_first')
                else:
                    pass

                data = (query_id, doc_id, BERT_encodings)
                # Append doc_id data topic
                if training_dataset:
                    if doc_id in self.qrels[query]:
                        self.topic_R_BERT_encodings.append(data)
                    else:
                        self.topic_N_BERT_encodings.append(data)
                else:
                    self.topic_BERT_encodings.append(data)

                # Store query as topic query.
                topic_query = query_id

            # Process any queries remaining.
        self.__process_topic(training_dataset=training_dataset)

        # write final chuck to file.
        self.__write_chuck_to_directory()


if __name__ == '__main__':

    index_path = '/Users/iain/LocalStorage/anserini_index/car_entity_v9'
    run_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'test.pages.cbor-hierarchical.entity.small.run')
    qrels_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'test.pages.cbor-hierarchical.entity.small.qrels')
    data_dir_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'small_dev')

    # index_path = '/nfs/trec_car/index/anserini_paragraphs/lucene-index.car17v2.0.paragraphsv2'
    # run_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmark_Y1_25.run'
    # qrels_path = '/nfs/trec_car/data/bert_reranker_datasets/dev_benchmark_Y1_25.qrels'
    # data_dir_path = '/nfs/trec_car/data/bert_reranker_datasets/'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 512
    use_token_type_ids = False
    processing = TrecCarProcessing(qrels_path=qrels_path,
                                   run_path=run_path,
                                   index_path=index_path,
                                   data_dir_path=data_dir_path,
                                   tokenizer=tokenizer,
                                   max_length=max_length)

    processing.build_dataset(training_dataset=True, chuck_topic_size=10)