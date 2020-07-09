
from learning.models import BertMultiTaskRanker
from learning.utils import BertDataset
from retrieval.tools import EvalTools, RetrievalUtils

from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch import nn

import collections
import torch
import datetime
import logging
import itertools
import time
import os


class FineTuningReRankingExperiments:

    eval_tools = EvalTools()
    # Evaluation config used in EvalTools.
    eval_config = [('map', None), ('Rprec', None), ('recip_rank', None), ('ndcg', 20), ('P', 20), ('recall', 40),
                   ('recall', 100), ('recall', 1000)]
    retrieval_utils = RetrievalUtils()
    pretrained_weights = 'bert-base-uncased'

    def __init__(self,
                 model_path=None,
                 dev_batch_size=None,
                 train_batch_size=None,
                 train_data_dir_path_passage=None,
                 train_data_dir_path_entity=None,
                 dev_data_dir_path_passage=None,
                 dev_data_dir_path_entity=None,
                 dev_qrels_path_passage=None,
                 dev_qrels_path_entity=None,
                 dev_run_path_passage=None,
                 dev_run_path_entity=None):

        # Load model from path or use pretrained_weights.
        self.model = self.__init_model(model_path=model_path)

        # Build PyTorch Dataloader for training data.
        self.train_dataloader_passage = self.__build_dataloader(data_dir_path=train_data_dir_path_passage,
                                                                batch_size=train_batch_size, random_sample=True)
        self.train_dataloader_entity = self.__build_dataloader(data_dir_path=train_data_dir_path_entity,
                                                                batch_size=train_batch_size, random_sample=True)

        # Build PyTorch Dataloader for validation data.
        self.dev_dataloader_passage = self.__build_dataloader(data_dir_path=dev_data_dir_path_passage,
                                                              batch_size=dev_batch_size, random_sample=False)
        self.dev_dataloader_entity = self.__build_dataloader(data_dir_path=dev_data_dir_path_entity,
                                                              batch_size=dev_batch_size, random_sample=False)
        # Store path of original dev/test run.
        self.dev_run_path_passage = dev_run_path_passage
        self.dev_run_path_entity = dev_run_path_entity

        # Store path of original dev/test qrels.
        self.dev_qrels_path_passage = dev_qrels_path_passage
        self.dev_qrels_path_entity = dev_qrels_path_entity

        # Dictionary from a qrels file. Key: query, value: list of relevant doc_ids.
        self.dev_qrels_passage = self.retrieval_utils.get_qrels_dict(qrels_path=dev_qrels_path_passage)
        self.dev_qrels_entity = self.retrieval_utils.get_qrels_dict(qrels_path=dev_qrels_path_entity)

        # List of tuples from run file (query, doc_id, R).
        self.dev_run_data_passage = self.__get_run_data(run_path=dev_run_path_passage, qrels=self.dev_qrels_passage)
        self.dev_run_data_entity = self.__get_run_data(run_path=dev_run_path_entity, qrels=self.dev_qrels_entity)

        # Store dev original labels. 1 = relevant, 0 = not relevant.
        self.dev_labels = None

        # Store dev logits between 0-1.
        self.dev_logits = None

        # Store torch device
        self.device = self.__get_torch_device()


    def __get_run_data(self, run_path, qrels):
        """ Reads run file returning list of tuples (query, doc_id, R) """
        if isinstance(run_path, str):
            run = []
            with open(run_path, 'r') as f_run:
                for line in f_run:
                    # Assumes run file is written in ascending order i.e. rank=1, rank=2, etc.
                    query, _, doc_id, _, _, _ = line.split()
                    if self.retrieval_utils.test_valid_line(line):
                        # Relevant
                        if doc_id in qrels[query]:
                            R = 1.0
                        # Not relevant.
                        else:
                            R = 0.0
                        run.append((query, doc_id, R))
            return run
        else:
            return None


    def __init_model(self, model_path):
        """ Initialise model with pre-trained weights or load from directory."""
        if model_path == None:
            return nn.DataParallel(BertMultiTaskRanker.from_pretrained(self.pretrained_weights))
        else:
            return nn.DataParallel(BertMultiTaskRanker.from_pretrained(model_path))


    def __build_dataloader(self, data_dir_path, batch_size, random_sample=False):
        """ Build PyTorch dataloader. """
        if (data_dir_path != None) and (batch_size != None):
            dataset = BertDataset(data_dir_path=data_dir_path)
            if random_sample:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
            return DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        else:
            return None


    def __format_time(self, elapsed):
        """ Formats time elapsed as hh:mm:ss """
        return str(datetime.timedelta(seconds=int(round((elapsed)))))


    def __flatten_list(self, l):
        """ Flattens list of lists to a single list. """
        return list(itertools.chain(*l))


    def __get_torch_device(self):
        """ get torch device to use (GPU and CPU). If GPU possible set model to use GPU i.e. model.cuda() """
        # Use GPUs if available.
        if torch.cuda.is_available():
            # Tell PyTorch to use the GPU.
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU: {}'.format(torch.cuda.get_device_name(0)))
            self.model.cuda()
            return torch.device("cuda")
        # Otherwise use CPU.
        else:
            print('No GPU available, using the CPU instead.')
            return torch.device("cpu")


    def __unpack_batch(self, batch):
        """ Unpack batch tensors (input_ids, token_type_ids, attention_mask, labels). """
        b_input_ids = batch[0].to(self.device)
        b_token_type_ids = batch[1].to(self.device)
        b_attention_mask = batch[2].to(self.device)
        b_labels = batch[3].to(self.device, dtype=torch.float)
        return b_input_ids, b_token_type_ids, b_attention_mask, b_labels


    def __update_dev_lables_and_logits(self, lables, logits):
        """ Update dev lables and logit lists from tensor. """
        if self.device == torch.device("cpu"):
            self.dev_labels += lables.cpu().numpy().tolist()
            self.dev_logits += self.__flatten_list(logits.cpu().detach().numpy().tolist())
        elif self.device == torch.device("cuda"):
            self.dev_labels += self.__flatten_list(lables.cpu().numpy().tolist())
            self.dev_logits += self.__flatten_list(logits.cpu().detach().numpy().tolist())
        else:
            print("NOT VALID DEVICE")
            raise


    def __get_bert_topic(self, original_topic, scores):
        """ Re-rank original topic based on BERT score. """
        bert_topic = []
        ordered_scores = sorted(list(set(scores)), reverse=True)
        for os in ordered_scores:
            ixs = [i for i, x in enumerate(scores) if x == os]
            for i in ixs:
                bert_topic.append(original_topic[i])
        return bert_topic


    def __add_topic_metrics(self, original_metrics_dict_sum, bert_metrics_dict_sum, oracle_metrics_dict_sum,
                             topic_query, original_topic, BERT_scores, dev_qrels):
        """ Add metrics of topic to dict sums. """
        # Get re-ranking BERT labels based on scores
        bert_topic = self.__get_bert_topic(original_topic=original_topic, scores=BERT_scores)
        # Get re-ranking oracle i.e. [0,1,0,0,1,1] -> [1,1,1,0,0,0]
        oracle_topic = sorted(original_topic, reverse=True)
        # Total number of relevant documents for query.
        R = len(dev_qrels[topic_query])

        # Get metrics
        _, original_metrics = self.eval_tools.get_query_metrics(run=original_topic, R=R, eval_config=self.eval_config)
        _, bert_metrics = self.eval_tools.get_query_metrics(run=bert_topic, R=R, eval_config=self.eval_config)
        _, oracle_metrics = self.eval_tools.get_query_metrics(run=oracle_topic, R=R, eval_config=self.eval_config)

        for k in original_metrics.keys():
            # Original metrics
            if k in original_metrics_dict_sum:
                original_metrics_dict_sum[k] += original_metrics[k]
            else:
                original_metrics_dict_sum[k] = original_metrics[k]
            # BERT metrics
            if k in bert_metrics_dict_sum:
                bert_metrics_dict_sum[k] += bert_metrics[k]
            else:
                bert_metrics_dict_sum[k] = bert_metrics[k]
            # Re-ranking oracle metrics
            if k in oracle_metrics_dict_sum:
                oracle_metrics_dict_sum[k] += oracle_metrics[k]
            else:
                oracle_metrics_dict_sum[k] = oracle_metrics[k]

        return original_metrics_dict_sum, bert_metrics_dict_sum, oracle_metrics_dict_sum


    def __log_eval_metrics(self, dev_qrels, dev_run_data):
        """ Log evaluation metrics (original, BERT re-rank based on score and re-ranking oracle). """
        # Assert validation outputs correct length.
        self.__assert_dev_lists_correct_lengths(dev_run_data)

        # Metric dicts for original run, re-ranking based on BERT scores, and re-ranking oracle.
        original_metrics_dict_sum = {}
        bert_metrics_dict_sum = {}
        oracle_metrics_dict_sum = {}

        # Store BERT score and original_topic (i.e. labels from original run).
        original_topic = []
        BERT_scores = []

        # Store topic query and count number of topics.
        topic_query = None
        topic_counter = 0

        for label, score, dev_run_data in zip(self.dev_labels, self.dev_logits, dev_run_data):
            # Unpack dev_run_data.
            query, doc_id, label_ground_truth = dev_run_data
            # Assert ordering looks correct.
            self.__assert_label_is_correct(label_ground_truth=label_ground_truth, label=label, query=query, doc_id=doc_id)

            if (topic_query != None) and (topic_query != query):
                # Add metrics of topic.
                original_metrics_dict_sum, bert_metrics_dict_sum, oracle_metrics_dict_sum = \
                    self.__add_topic_metrics(original_metrics_dict_sum=original_metrics_dict_sum,
                                             bert_metrics_dict_sum=bert_metrics_dict_sum,
                                             oracle_metrics_dict_sum=oracle_metrics_dict_sum,
                                             topic_query=topic_query,
                                             original_topic=original_topic,
                                             BERT_scores=BERT_scores,
                                             dev_qrels=dev_qrels)
                topic_counter += 1
                # Start new topic run.
                original_topic = []
                BERT_scores = []

            # Update topic run.
            topic_query = query
            original_topic.append(label_ground_truth)
            BERT_scores.append(score)

        # Add metrics of final topic.
        if len(original_topic) > 0:
            original_metrics_dict_sum, bert_metrics_dict_sum, oracle_metrics_dict_sum = \
                self.__add_topic_metrics(original_metrics_dict_sum=original_metrics_dict_sum,
                                         bert_metrics_dict_sum=bert_metrics_dict_sum,
                                         oracle_metrics_dict_sum=oracle_metrics_dict_sum,
                                         topic_query=topic_query,
                                         original_topic=original_topic,
                                         BERT_scores=BERT_scores,
                                         dev_qrels=dev_qrels)
            topic_counter += 1

        # Mean metric dicts by dividing each metric by topic_counter.
        original_metrics_dict = {}
        bert_metrics_dict = {}
        oracle_metrics_dict = {}
        for k in original_metrics_dict_sum:
            original_metrics_dict[k] = original_metrics_dict_sum[k] / topic_counter
            bert_metrics_dict[k] = bert_metrics_dict_sum[k] / topic_counter
            oracle_metrics_dict[k] = oracle_metrics_dict_sum[k] / topic_counter

        # Log metrics.
        logging.info('Original: \t{}'.format(original_metrics_dict))
        logging.info('BERT:     \t{}'.format(bert_metrics_dict))
        logging.info('Oracle:   \t{}'.format(oracle_metrics_dict))


    def __validation_run(self, head_flag, dev_dataloader):
        """ Run validation to build dev_labels/dev_logits and return average dev loss. ."""
        # Dev loss counter.
        dev_loss = 0
        # Total number of dev batches.
        num_dev_steps = len(dev_dataloader)

        # Set model to evaluation mode i.e. not weight updates.
        self.model.eval()

        # Store prediction logits and labels in lists.
        self.dev_labels = []
        self.dev_logits = []

        for dev_step, dev_batch in enumerate(dev_dataloader):
            # Unpack batch (input_ids, token_type_ids, attention_mask, labels).
            b_input_ids, b_token_type_ids, b_attention_mask, b_labels = self.__unpack_batch(batch=dev_batch)
            # With no gradients
            with torch.no_grad():
                loss, logits = self.model.forward(head_flag=head_flag,
                                                  input_ids=b_input_ids,
                                                  token_type_ids=b_token_type_ids,
                                                  attention_mask=b_attention_mask,
                                                  labels=b_labels)
            # Update dev loss counter.
            dev_loss += loss.sum().item()

            # Update list of dev lables and logits
            self.__update_dev_lables_and_logits(lables=b_labels, logits=logits)

        # Report the final accuracy for this validation run.
        return dev_loss / num_dev_steps


    def run_experiment_single_head(self, head_flag, epochs=1, lr=2e-5, eps=1e-8, weight_decay=0.01,
                                   warmup_percentage=0.1, experiments_dir=None, experiment_name=None, logging_steps=100):
        """ Run training and validation for a single head. """
        assert head_flag == 'passage' or head_flag == 'entity'
        if head_flag == 'passage':
            train_dataloader = self.train_dataloader_passage
            dev_dataloader = self.dev_dataloader_passage
            dev_qrels = self.dev_qrels_passage
            dev_run_data = self.dev_run_data_passage
        else:
            train_dataloader = self.train_dataloader_entity
            dev_dataloader = self.dev_dataloader_entity
            dev_qrels = self.dev_qrels_entity
            dev_run_data = self.dev_run_data_entity

        # Define experiment_path directory to contain all logging, models and results.
        experiment_path = os.path.join(experiments_dir, experiment_name)

        # Make experiment_path if does not already exist.
        if os.path.isdir(experiment_path) == False:
            os.mkdir(experiment_path)

        # Start logging.
        logging_path = os.path.join(experiment_path, 'output.log')
        print('Starting logging: {}'.format(logging_path))
        logging.basicConfig(filename=logging_path, level=logging.DEBUG)

        # Loop over epochs (1 -> epochs).
        for epoch_i in range(1, epochs + 1):

            logging.info("=================================")
            logging.info('======== Epoch {:} / {:} ========'.format(epoch_i, epochs))
            logging.info("=================================")

            # ========================================
            #               Training
            # ========================================

            # Initialise optimizer and scheduler for training.
            optimizer = AdamW(self.model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
            num_train_steps = len(train_dataloader)
            num_warmup_steps = int(num_train_steps * warmup_percentage)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                        num_training_steps=num_train_steps)

            # Time beginning training.
            train_start_time = time.time()
            # Set model in training mode.
            self.model.train()
            # Train loss counter.
            train_loss = 0

            for train_step, train_batch in enumerate(train_dataloader):
                # Set gradient to zero.
                self.model.zero_grad()
                # Unpack batch (input_ids, token_type_ids, attention_mask, labels).
                b_input_ids, b_token_type_ids, b_attention_mask, b_labels = self.__unpack_batch(batch=train_batch)
                # Forward pass to retrieve
                loss, logits = self.model.forward(head_flag=head_flag,
                                                  input_ids=b_input_ids,
                                                  attention_mask=b_attention_mask,
                                                  token_type_ids=b_token_type_ids,
                                                  labels=b_labels)

                # Add loss to train loss counter
                train_loss += loss.sum().item()
                # Backpropogate loss.
                loss.sum().backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Next step of optimizer and scheduler.
                optimizer.step()
                scheduler.step()

                # Progress update every X batches or last batch (logging and validation).
                if ((train_step + 1) % logging_steps == 0) or ((train_step + 1) == num_train_steps):

                    # Calculate average training loss
                    avg_train_loss = train_loss / (train_step + 1)

                    logging.info('----- Epoch {} / Batch {} -----\n'.format(str(epoch_i), str(train_step + 1)))
                    logging.info("Training loss: {0:.5f}".format(avg_train_loss))
                    logging.info("Training time: {:}".format(self.__format_time(time.time() - train_start_time)))

                    # ========================================
                    #               Validation
                    # ========================================

                    # Dev beginning training.
                    dev_start_time = time.time()

                    av_dev_loss = self.__validation_run(head_flag=head_flag, dev_dataloader=dev_dataloader)

                    logging.info("Validation loss: {0:.5f}".format(av_dev_loss))
                    logging.info("Validation time: {:}".format(self.__format_time(time.time() - dev_start_time)))

                    self.__log_eval_metrics(dev_qrels=dev_qrels, dev_run_data=dev_run_data)

                    # Save model and weights to directory.
                    model_dir = os.path.join(experiment_path, 'epoch{}_batch{}/'.format(epoch_i, train_step + 1))
                    if os.path.isdir(model_dir) == False:
                        os.mkdir(model_dir)
                    try:
                        self.model.module.save_pretrained(model_dir)
                    except AttributeError:
                        self.model.save_pretrained(model_dir)


    def run_experiment_multi_head(self, epochs=1, lr=2e-5, eps=1e-8, weight_decay=0.01, warmup_percentage=0.1,
                                  experiments_dir=None, experiment_name=None, logging_steps=100):
        """ Run training and validation for a multi head. """
        # Define experiment_path directory to contain all logging, models and results.
        experiment_path = os.path.join(experiments_dir, experiment_name)

        # Make experiment_path if does not already exist.
        if os.path.isdir(experiment_path) == False:
            os.mkdir(experiment_path)

        # Start logging.
        logging_path = os.path.join(experiment_path, 'output.log')
        print('Starting logging: {}'.format(logging_path))
        logging.basicConfig(filename=logging_path, level=logging.DEBUG)

        # Loop over epochs (1 -> epochs).
        for epoch_i in range(1, epochs + 1):

            logging.info("=================================")
            logging.info('======== Epoch {:} / {:} ========'.format(epoch_i, epochs))
            logging.info("=================================")

            # ========================================
            #               Training
            # ========================================

            # Initialise optimizer and scheduler for training.
            optimizer = AdamW(self.model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
            num_train_steps = min(len(self.train_dataloader_passage), len(self.train_dataloader_entity))
            num_warmup_steps = int(num_train_steps * warmup_percentage)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                        num_training_steps=num_train_steps)

            # Time beginning training.
            train_start_time = time.time()
            # Set model in training mode.
            self.model.train()
            # Train loss counters.
            train_loss_passage = 0
            train_loss_entity = 0

            train_step = 0
            for train_batch_passage, train_batch_entity in zip(self.train_dataloader_passage, self.train_dataloader_entity):
                # Feedforward both heads
                for head_flag, train_batch in zip(['passage', 'entity'], [train_batch_passage, train_batch_entity]):
                    # Set gradient to zero.
                    self.model.zero_grad()
                    # Unpack batch (input_ids, token_type_ids, attention_mask, labels).
                    b_input_ids, b_token_type_ids, b_attention_mask, b_labels = self.__unpack_batch(batch=train_batch_passage)
                    # Forward pass to retrieve
                    loss, logits = self.model.forward(head_flag=head_flag,
                                                      input_ids=b_input_ids,
                                                      attention_mask=b_attention_mask,
                                                      token_type_ids=b_token_type_ids,
                                                      labels=b_labels)

                    # Add loss to train loss counter
                    if head_flag == 'passage':
                        print('== passage ==')
                        print(type(loss))
                        print(loss)

                        loss_passage = loss
                        train_loss_passage += loss.sum().item()
                    else:
                        print('== entity ==')
                        print(type(loss))
                        print(loss)
                        loss_entity = loss
                        train_loss_entity += loss.sum().item()

                # Backpropogate loss.
                print('== total_loss ==')
                loss_total = loss_passage + loss_entity
                print(type(loss_total))
                print(loss_total)
                print('== total_loss.sum() ==')
                print(loss_total.sum())
                loss_total.sum().backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Next step of optimizer and scheduler.
                optimizer.step()
                scheduler.step()

                # Progress update every X batches or last batch (logging and validation).
                if ((train_step + 1) % logging_steps == 0) or ((train_step + 1) == num_train_steps):

                    # Calculate average training loss
                    avg_train_loss_passage = train_loss_passage / (train_step + 1)
                    avg_train_loss_entity = train_loss_entity / (train_step + 1)

                    logging.info('----- Epoch {} / Batch {} -----\n'.format(str(epoch_i), str(train_step + 1)))
                    logging.info("Training loss passage: {0:.5f}".format(avg_train_loss_passage))
                    logging.info("Training loss entity: {0:.5f}".format(avg_train_loss_entity))
                    logging.info("Training time: {:}".format(self.__format_time(time.time() - train_start_time)))

                    # ========================================
                    #               Validation
                    # ========================================

                    for head_flag in ['passage', 'entity']:
                        if head_flag == 'passage':
                            dev_dataloader = self.dev_dataloader_passage
                            dev_qrels = self.dev_qrels_passage
                            dev_run_data = self.dev_run_data_passage
                        else:
                            dev_dataloader = self.dev_dataloader_entity
                            dev_qrels = self.dev_qrels_entity
                            dev_run_data = self.dev_run_data_entity

                        # Dev beginning training.
                        dev_start_time = time.time()

                        av_dev_loss = self.__validation_run(head_flag=head_flag, dev_dataloader=dev_dataloader)

                        logging.info("=== Validation {} ===".format(head_flag))
                        logging.info("Validation loss: {0:.5f}".format(av_dev_loss))
                        logging.info("Validation time: {:}".format(self.__format_time(time.time() - dev_start_time)))

                        self.__log_eval_metrics(dev_qrels=dev_qrels, dev_run_data=dev_run_data)

                    # Save model and weights to directory.
                    model_dir = os.path.join(experiment_path, 'epoch{}_batch{}/'.format(epoch_i, train_step + 1))
                    if os.path.isdir(model_dir) == False:
                        os.mkdir(model_dir)
                    try:
                        self.model.module.save_pretrained(model_dir)
                    except AttributeError:
                        self.model.save_pretrained(model_dir)

                train_step += 1


    def __write_topic_to_file(self, rerank_run_path, doc_ids, query, scores):
        """ Write topic to run file. """
        with open(rerank_run_path, "a+") as f_run:
            # Build dict of {doc_ids: score}
            d = {i[0]: i[1] for i in zip(doc_ids, scores)}
            # Order dict based on score
            od = collections.OrderedDict(sorted(d.items(), key=lambda item: item[1], reverse=True))
            rank = 1
            # Writing to file in ascending rank.
            for doc_id, score in od.items():
                f_run.write(" ".join((query, "Q0", str(doc_id), str(rank), "{:.6f}".format(score), "BERT")) + '\n')
                rank += 1


    def __assert_dev_lists_correct_lengths(self, dev_run_data):
        """ Assert dev_labels and dev_logits from validation are correct length i.e. length of original run file. """
        assert len(self.dev_labels) == len(self.dev_logits) == len(dev_run_data),\
            "dev_labels len: {}, dev_logits len: {}, dev_run_data: {}".format(len(self.dev_labels), len(self.dev_logits), len(dev_run_data))


    def __assert_label_is_correct(self, label_ground_truth, label, query, doc_id):
        """ Assert label from dataset (0,1) is the same as the original label.  Tests data ordering is consistent. """
        if self.device == torch.device("cpu"):
            assert label_ground_truth == label[0], "label_ground_truth: {} vs. label: {} -> query: {}, doc_id: {}".format(label_ground_truth, label[0], query, doc_id)
        elif self.device == torch.device("cuda"):
            assert label_ground_truth == label, "label_ground_truth: {} vs. label: {} -> query: {}, doc_id: {}".format(label_ground_truth, label, query, doc_id)
        else:
            print("NOT VALID DEVICE")
            raise


    def write_rerank_to_run_file(self, rerank_run_path, dev_run_data):
        """ Process re-ranking from validation run and write to TREC run file. """
        # Assert validation outputs correct length.
        self.__assert_dev_lists_correct_lengths(dev_run_data=dev_run_data)

        # Store BERT score and doc_id for each topic run.
        BERT_scores = []
        doc_ids = []

        # Store topic query.
        topic_query = None

        # Loop over dev_labels, dev_logits and dev_run_data.
        for label, score, dev_run_data in zip(self.dev_labels, self.dev_logits, dev_run_data):
            # Unpack dev_run_data.
            query, doc_id, label_ground_truth = dev_run_data
            # Assert ordering looks correct.
            self.__assert_label_is_correct(label_ground_truth=label_ground_truth, label=label, query=query, doc_id=doc_id)

            if (topic_query != None) and (topic_query != query):
                # End of topic run -> write to file.
                self.__write_topic_to_file(rerank_run_path=rerank_run_path, query=topic_query, doc_ids=doc_ids,
                                           scores=BERT_scores)

                # Start new topic run.
                BERT_scores = []
                doc_ids = []

            # Update topic run.
            topic_query = query
            BERT_scores.append(score)
            doc_ids.append(doc_id)

        # Write final topic run.
        if len(doc_ids) > 0:
            self.__write_topic_to_file(rerank_run_path=rerank_run_path, query=topic_query, doc_ids=doc_ids,
                                       scores=BERT_scores)


    def inference(self, head_flag, rerank_run_path, do_eval=True):
        """ Run inference and produce BERT re-ranking run and evaluation. """
        assert head_flag == 'passage' or head_flag == 'entity'
        if head_flag == 'passage':
            dev_dataloader = self.dev_dataloader_passage
            dev_qrels_path = self.dev_qrels_path_passage
            dev_run_data = self.dev_run_data_passage
        else:
            dev_dataloader = self.dev_dataloader_entity
            dev_qrels_path = self.dev_qrels_path_entity
            dev_run_data = self.dev_run_data_entity

        # Run Validation.
        self.__validation_run(head_flag=head_flag, dev_dataloader=dev_dataloader)

        # Write re-ranking run
        self.write_rerank_to_run_file(rerank_run_path=rerank_run_path, dev_run_data=dev_run_data)

        # If 'do_eval' write eval files by query and overall.
        if do_eval:
            self.eval_tools.write_eval_from_qrels_and_run(qrels_path=dev_qrels_path,
                                                          run_path=rerank_run_path,
                                                          eval_config=self.eval_config)


if __name__ == '__main__':
    pass