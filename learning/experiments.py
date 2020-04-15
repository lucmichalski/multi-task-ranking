
from learning.models import BertMultiTaskRanker
from learning.utils import BertDataset
from retrieval.tools import EvalTools

from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, SequentialSampler
from torch import nn

import collections
import numpy as np
import torch
import random
import datetime
import logging
import itertools
import time
import os


class FineTuningReRankingExperiments:

    pretrained_weights = 'bert-base-uncased'

    def __init__(self, model_path=None, train_data_dir_path=None, train_batch_size=None, dev_data_dir_path=None, dev_batch_size=None,
                 dev_qrels_path=None, dev_run_path=None):
        self.model = self.__init_model(model_path=model_path)
        self.eval_tools = EvalTools()
        self.train_dataloader = self.__build_dataloader(data_dir_path=train_data_dir_path, batch_size=train_batch_size)
        self.dev_dataloader = self.__build_dataloader(data_dir_path=dev_data_dir_path, batch_size=dev_batch_size)
        self.dev_run_path = dev_run_path
        self.dev_qrels_path = dev_qrels_path
        self.dev_qrels = self.__get_qrels(qrels_path=dev_qrels_path)
        self.dev_run_data = self.__get_run_data(run_path=dev_run_path)
        self.dev_labels = None
        self.dev_logits = None
        self.device = self.__get_torch_device()
        self.eval_config = {
            'map': {'k': None},
            'Rprec': {'k': None},
            'recip_rank': {'k': None},
            'P': {'k': 20},
            'recall': {'k': 40},
            'ndcg': {'k': 20},
        }


    def __get_qrels(self, qrels_path):
        """ """
        return self.eval_tools.get_qrels_dict(qrels_path=qrels_path)


    def __get_run_data(self, run_path):
        """ """
        run = []
        with open(run_path, 'r') as f_run:
            for line in f_run:
                # Assumes run file is written in ascending order i.e. rank=1, rank=2, etc.
                query, _, doc_id, _, _, _ = line.split()
                if doc_id in self.dev_qrels[query]:
                    R = 1.0
                else:
                    R = 0.0
                run.append((query, doc_id, R))
        return run


    def __init_model(self, model_path):
        """ Initialise model with pre-trained weights."""
        if model_path == None:
            return nn.DataParallel(BertMultiTaskRanker.from_pretrained(self.pretrained_weights))
        else:
            return nn.DataParallel(BertMultiTaskRanker.from_pretrained(model_path))


    def __build_dataloader(self, data_dir_path, batch_size):
        """ Build PyTorch dataloader. """
        if (data_dir_path != None) and (batch_size != None):
            dataset = BertDataset(data_dir_path=data_dir_path)
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
        """ . """
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
        """ """
        bert_topic = []
        ordered_scores = sorted(list(set(scores)), reverse=True)
        for os in ordered_scores:
            ixs = [i for i, x in enumerate(scores) if x == os]
            for i in ixs:
                bert_topic.append(original_topic[i])
        return bert_topic


    def __get_topics_metrics(self, original_metrics_dict_sum, bert_metrics_dict_sum, oracle_metrics_dict_sum,
                             topic_query, original_topic, BERT_scores, eval_config):
        """ """
        bert_topic = self.__get_bert_topic(original_topic=original_topic, scores=BERT_scores)
        oracle_topic = sorted(original_topic, reverse=True)

        R = len(self.dev_qrels[topic_query])

        _, original_metrics = self.eval_tools.get_query_metrics(run=original_topic, R=R, eval_config=eval_config)
        _, bert_metrics = self.eval_tools.get_query_metrics(run=bert_topic, R=R, eval_config=eval_config)
        _, oracle_metrics = self.eval_tools.get_query_metrics(run=oracle_topic, R=R, eval_config=eval_config)

        for k in original_metrics.keys():

            if k in original_metrics_dict_sum:
                original_metrics_dict_sum[k] += original_metrics[k]
            else:
                original_metrics_dict_sum[k] = original_metrics[k]

            if k in bert_metrics_dict_sum:
                bert_metrics_dict_sum[k] += bert_metrics[k]
            else:
                bert_metrics_dict_sum[k] = bert_metrics[k]

            if k in oracle_metrics_dict_sum:
                oracle_metrics_dict_sum[k] += oracle_metrics[k]
            else:
                oracle_metrics_dict_sum[k] = oracle_metrics[k]

        return original_metrics_dict_sum, bert_metrics_dict_sum, oracle_metrics_dict_sum


    def __get_eval_metrics(self):
        """ """
        assert len(self.dev_labels) == len(self.dev_logits) == len(self.dev_run_data), \
            "dev_labels len: {}, dev_logits len: {}, dev_run_data: {}".format(
                len(self.dev_labels), len(self.dev_logits),len(self.dev_run_data))

        original_metrics_dict_sum = {}
        bert_metrics_dict_sum = {}
        oracle_metrics_dict_sum = {}

        original_topic = []
        BERT_scores = []

        topic_query = None
        topic_counter = 0

        for label, score, dev_run_data in zip(self.dev_labels, self.dev_logits, self.dev_run_data):
            query, doc_id, label_ground_truth = dev_run_data
            #assert label_ground_truth == label[0], "label_ground_truth: {} vs. label: {}".format(label_ground_truth, label[0])
            assert label_ground_truth == label, "label_ground_truth: {} vs. label: {}".format(label_ground_truth, label)

            if (topic_query != None) and (topic_query != query):
                # get topics of metrics
                topic_counter += 1

                original_metrics_dict_sum, bert_metrics_dict_sum, oracle_metrics_dict_sum = \
                    self.__get_topics_metrics(original_metrics_dict_sum=original_metrics_dict_sum,
                                              bert_metrics_dict_sum=bert_metrics_dict_sum,
                                              oracle_metrics_dict_sum=oracle_metrics_dict_sum,
                                              topic_query=topic_query,
                                              original_topic=original_topic,
                                              BERT_scores=BERT_scores,
                                              eval_config=self.eval_config)

                original_topic = []
                BERT_scores = []

            topic_query = query
            original_topic.append(label_ground_truth)
            BERT_scores.append(score)

        if len(original_topic) > 0:

            topic_counter += 1

            original_metrics_dict_sum, bert_metrics_dict_sum, oracle_metrics_dict_sum = \
                self.__get_topics_metrics(original_metrics_dict_sum=original_metrics_dict_sum,
                                          bert_metrics_dict_sum=bert_metrics_dict_sum,
                                          oracle_metrics_dict_sum=oracle_metrics_dict_sum,
                                          topic_query=topic_query,
                                          original_topic=original_topic,
                                          BERT_scores=BERT_scores,
                                          eval_config=self.eval_config)

        original_metrics_dict = {}
        bert_metrics_dict = {}
        oracle_metrics_dict = {}
        for k in original_metrics_dict_sum:
            original_metrics_dict[k] = original_metrics_dict_sum[k] / topic_counter
            bert_metrics_dict[k] = bert_metrics_dict_sum[k] / topic_counter
            oracle_metrics_dict[k] = oracle_metrics_dict_sum[k] / topic_counter

        logging.info('Original: \t{}'.format(original_metrics_dict))
        logging.info('BERT:     \t{}'.format(bert_metrics_dict))
        logging.info('Oracle:   \t{}'.format(oracle_metrics_dict))


    def __validation_run(self, head_flag):
        """ """
        # Dev loss counter.
        dev_loss = 0
        # Total number of dev batches.
        num_dev_steps = len(self.dev_dataloader)

        # Set model to evaluation mode i.e. not weight updates.
        self.model.eval()

        # Store prediction logits and labels in lists.
        self.dev_labels = []
        self.dev_logits = []

        for dev_step, dev_batch in enumerate(self.dev_dataloader):
            # Unpack batch (input_ids, token_type_ids, attention_mask, labels).
            b_input_ids, b_token_type_ids, b_attention_mask, b_labels = self.__unpack_batch(
                batch=dev_batch)

            # With no gradients
            with torch.no_grad():
                loss, logits = self.model.module.forward_head(head_flag=head_flag,
                                                              input_ids=b_input_ids,
                                                              token_type_ids=b_token_type_ids,
                                                              attention_mask=b_attention_mask,
                                                              labels=b_labels)
            # Update dev loss counter.
            dev_loss += loss.mean().item()

            # Update list of dev lables and logits
            self.__update_dev_lables_and_logits(lables=b_labels, logits=logits)

        # Report the final accuracy for this validation run.
        return dev_loss / num_dev_steps


    def run_experiment_single_head(self, head_flag='passage', epochs=1, lr=2e-5, eps=1e-8, weight_decay=0.01,
                                   num_warmup_steps=0, experiments_dir=None, experiment_name=None, logging_steps=100):
        """ """
        # Define experiment_path directory to contain all logging, models and results.
        experiment_path = os.path.join(experiments_dir, experiment_name)

        # Make experiment_path if does not already exist.
        if os.path.isdir(experiment_path) == False:
            os.mkdir(experiment_path)

        # Logging.
        logging_path = os.path.join(experiment_path, 'output.log')
        print('Starting logging: {}'.format(logging_path))
        logging.basicConfig(filename=logging_path, level=logging.DEBUG)

        # Initialise optimizer and scheduler for training.
        optimizer = AdamW(self.model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
        num_train_steps = len(self.train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_steps)

        # Loop over epochs (1 -> epochs).
        for epoch_i in range(1, epochs + 1):

            logging.info("=================================")
            logging.info('======== Epoch {:} / {:} ========'.format(epoch_i, epochs))
            logging.info("=================================")

            # ========================================
            #               Training
            # ========================================

            # Time beginning training.
            train_start_time = time.time()
            # Set model in training mode.
            self.model.train()
            # Train loss counter.
            train_loss = 0

            for train_step, train_batch in enumerate(self.train_dataloader):
                # Unpack batch (input_ids, token_type_ids, attention_mask, labels).
                b_input_ids, b_token_type_ids, b_attention_mask, b_labels = self.__unpack_batch(batch=train_batch)

                # Set gradient to zero.
                self.model.zero_grad()
                # Forward pass to retrieve
                loss, logits = self.model.module.forward_head(head_flag=head_flag,
                                                              input_ids=b_input_ids,
                                                              attention_mask=b_attention_mask,
                                                              token_type_ids=b_token_type_ids,
                                                              labels=b_labels)
                # Add loss to train loss counter
                train_loss += loss.mean().item()
                # Backpropogate loss.
                loss.mean().backward()
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

                    av_dev_loss = self.__validation_run(head_flag=head_flag)

                    logging.info("Validation loss: {0:.5f}".format(av_dev_loss))
                    logging.info("Validation time: {:}".format(self.__format_time(time.time() - dev_start_time)))

                    self.__get_eval_metrics()

                    # Save model and weights to directory.
                    model_dir = os.path.join(experiment_path, 'epoch{}_batch{}/'.format(epoch_i, train_step + 1))
                    if os.path.isdir(model_dir) == False:
                        os.mkdir(model_dir)
                    try:
                        self.model.module.save_pretrained(model_dir)
                    except AttributeError:
                        self.save_pretrained(model_dir)


    def run_grid_search(self):
        """ """
        pass


    def __write_to_file(self, rerank_run_path, doc_ids, query, scores):

        with open(rerank_run_path, "a+") as f_run:
            d = {i[0]: i[1] for i in zip(doc_ids, scores)}
            od = collections.OrderedDict(sorted(d.items(), key=lambda item: item[1], reverse=True))
            rank = 1
            for doc_id in od.keys():
                output_line = " ".join((query, "Q0", str(doc_id), str(rank), str(od[doc_id]), "BERT")) + '\n'
                f_run.write(output_line)
                rank += 1


    def write_rerank_run(self, rerank_run_path):
        """ """
        assert len(self.dev_labels) == len(self.dev_logits) == len(self.dev_run_data), \
            "dev_labels len: {}, dev_logits len: {}, dev_run_data: {}".format(
                len(self.dev_labels), len(self.dev_logits), len(self.dev_run_data))

        original_topic = []
        BERT_scores = []
        doc_ids = []

        topic_query = None

        for label, score, dev_run_data in zip(self.dev_labels, self.dev_logits, self.dev_run_data):
            query, doc_id, label_ground_truth = dev_run_data
            #assert label_ground_truth == label[0], "label_ground_truth: {} vs. label: {}".format(label_ground_truth, label[0])
            assert label_ground_truth == label, "label_ground_truth: {} vs. label: {}".format(label_ground_truth, label)

            if (topic_query != None) and (topic_query != query):
                # get topics of metrics

                self.__write_to_file(rerank_run_path=rerank_run_path, query=topic_query, doc_ids=doc_ids, scores=BERT_scores)

                original_topic = []
                BERT_scores = []

            topic_query = query
            original_topic.append(label_ground_truth)
            BERT_scores.append(score)
            doc_ids.append(doc_id)

            print(label[0], score, query, doc_id, label_ground_truth)

        if len(original_topic) > 0:
            self.__write_to_file(rerank_run_path=rerank_run_path, query=topic_query, doc_ids=doc_ids, scores=BERT_scores)

    def inference(self, head_flag, rerank_run_path, do_eval=True):
        """ """

        self.__validation_run(head_flag=head_flag)

        self.write_rerank_run(rerank_run_path)

        self.eval_tools.write_eval_from_qrels_and_run(qrels_path=self.dev_qrels_path,
                                                      run_path=rerank_run_path,
                                                      eval_config=self.eval_config)




if __name__ == '__main__':
    train_data_dir_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'small_train')
    train_batch_size = 2
    dev_data_dir_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'small_dev')
    dev_batch_size = 2
    dev_qrels_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'test.pages.cbor-hierarchical.entity.small.qrels')
    dev_run_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'test.pages.cbor-hierarchical.entity.small.run')

    experiment = FineTuningReRankingExperiments(train_data_dir_path=train_data_dir_path,
                                                train_batch_size=train_batch_size,
                                                dev_data_dir_path=dev_data_dir_path,
                                                dev_batch_size=dev_batch_size,
                                                dev_qrels_path=dev_qrels_path,
                                                dev_run_path=dev_run_path)

    # epochs = 1
    # lr = 5e-5
    # eps = 1e-8
    # weight_decay = 0.01
    # num_warmup_steps = 0
    # seed_val = 42
    # experiments_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'exp')
    # experiment_name = 'test_exp_2'
    # logging_steps = 10
    #
    # experiment.run_experiment_single_head(epochs=epochs,
    #                                       lr=lr,
    #                                       eps=eps,
    #                                       weight_decay=weight_decay,
    #                                       num_warmup_steps=num_warmup_steps,
    #                                       experiments_dir=experiments_dir,
    #                                       experiment_name=experiment_name,
    #                                       logging_steps=logging_steps)

    rerank_run_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'rerank.run')
    experiment.inference(head_flag='passage', rerank_run_path=rerank_run_path)
