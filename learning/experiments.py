
from learning.models import BertMultiTaskRanker
from learning.utils import BertDataset

from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, SequentialSampler
from torch import nn

import numpy as np
import torch
import random
import datetime
import logging
import itertools
import time
import os

class FineTuningReRanking:

    pretrained_weights = 'bert-base-uncased'

    def __init__(self, train_data_dir_path, train_batch_size, dev_data_dir_path, dev_batch_size):
        self.model = nn.DataParallel(BertMultiTaskRanker.from_pretrained(self.pretrained_weights))
        self.train_dataloader = self.__build_dataloader(data_dir_path=train_data_dir_path, batch_size=train_batch_size)
        self.dev_dataloader = self.__build_dataloader(data_dir_path=dev_data_dir_path, batch_size=dev_batch_size)


    def __build_dataloader(self, data_dir_path, batch_size):
        """ Build PyTorch dataloader. """
        dataset = BertDataset(data_dir_path=data_dir_path)
        sampler = SequentialSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=batch_size)


    def __log_parameters(self):
        logging.info('--- EXPERIMENT PARAMETERS ---')
        # setup_strings = ['epochs', 'lr', 'eps', 'weight_decay', 'num_warmup_steps', 'seed_val', 'write', 'exp_dir',
        #                  'experiment_name', 'do_eval', 'logging_steps', 'run_path', 'qrels_path']
        # setup_values = [epochs, lr, eps, weight_decay, num_warmup_steps, seed_val, write, exp_dir, experiment_name,
        #                 do_eval, logging_steps, run_path, qrels_path]
        # for i in zip(setup_strings, setup_values):
        #     logging.info('{}: {}'.format(i[0], i[1]))
        logging.info('-----------------------------')


    def __format_time(self, elapsed):
        """ Formats time elapsed as hh:mm:ss """
        return str(datetime.timedelta(seconds=int(round((elapsed)))))


    def __flatten_list(self, l):
        """ Flattens list of lists to a single list. """
        return list(itertools.chain(*l))


    def run_experiment(self, epochs=1, lr=2e-5, eps=1e-8, weight_decay=0.01, num_warmup_steps=0, seed_val=42, experiments_dir=None, experiment_name=None, write=False, logging_steps=100):
        """ """
        # Set the seed value all over the place to make this reproducible.
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # Define experiment_path directory to contain all logging, models and results.
        experiment_path = os.path.join(experiments_dir, experiment_name)

        if write:
            # Make experiment_path if does not already exist.
            if os.path.isdir(experiment_path) == False:
                os.mkdir(experiment_path)
            print('*** Starting logging ***')
            logging_path = os.path.join(experiment_path, 'output.log')
            logging.basicConfig(filename=logging_path, level=logging.DEBUG)

        # Use GPUs if available.
        if torch.cuda.is_available():
            # Tell PyTorch to use the GPU.
            device = torch.device("cuda")
            logging.info('There are %d GPU(s) available.' % torch.cuda.device_count())
            logging.info('We will use the GPU: {}'.format(torch.cuda.get_device_name(0)))
            self.model.cuda()
        # Otherwise use CPU.
        else:
            logging.info('No GPU available, using the CPU instead.')
            device = torch.device("cpu")

        # Log experiment parameters
        self.__log_parameters()

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
                b_input_ids = train_batch[0].to(device)
                b_token_type_ids = train_batch[1].to(device)
                b_attention_mask = train_batch[2].to(device)
                b_labels = train_batch[3].to(device, dtype=torch.float)

                # Set gradient to zero.
                self.model.zero_grad()
                # Forward pass to retrieve
                loss, logits = self.model.module.forward_passage(input_ids=b_input_ids,
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
                    # Dev loss counter.
                    dev_loss = 0
                    # Total number of dev batches.
                    num_dev_steps = len(self.dev_dataloader)

                    # pred_list = []
                    # label_list = []

                    # Set model to evaluation mode i.e. not weight updates.
                    self.model.eval()

                    for dev_step, dev_batch in enumerate(self.dev_dataloader):

                        # Unpack batch (input_ids, token_type_ids, attention_mask, labels).
                        b_input_ids = dev_batch[0].to(device)
                        b_token_type_ids = dev_batch[1].to(device)
                        b_attention_mask = dev_batch[2].to(device)
                        b_labels = dev_batch[3].to(device, dtype=torch.float)

                        # With no gradients
                        with torch.no_grad():
                            loss, logits = self.model.module.forward_passage(input_ids=b_input_ids,
                                                                             token_type_ids=b_token_type_ids,
                                                                             attention_mask=b_attention_mask,
                                                                             labels=b_labels)
                        # Update dev loss counter.
                        dev_loss += loss.mean().item()

                        # if device == torch.device("cpu"):
                        #     pred_list += self.__flatten_list(outputs[1].cpu().detach().numpy().tolist())
                        #     label_list += dev_batch[3].cpu().numpy().tolist()
                        # else:
                        #     pred_list += self.__flatten_list(outputs[1].cpu().detach().numpy().tolist())
                        #     label_list += self.__flatten_list(b_labels.cpu().numpy().tolist())

                    # Report the final accuracy for this validation run.
                    av_dev_loss = dev_loss / num_dev_steps
                    logging.info("Validation loss: {0:.5f}".format(av_dev_loss))
                    logging.info("Validation time: {:}".format(self.__format_time(time.time() - dev_start_time)))


if __name__ == '__main__':
    train_data_dir_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'results')
    train_batch_size = 1
    dev_data_dir_path = train_data_dir_path
    dev_batch_size = 1

    experiment = FineTuningReRanking(train_data_dir_path=train_data_dir_path,
                                     train_batch_size=train_batch_size,
                                     dev_data_dir_path=dev_data_dir_path,
                                     dev_batch_size=dev_batch_size)

    epochs = 1
    lr = 2e-5
    eps = 1e-8
    weight_decay = 0.01
    num_warmup_steps = 0
    seed_val = 42
    experiments_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data')
    experiment_name = 'test_exp_1'
    write = True
    logging_steps = 100

    experiment.run_experiment(epochs=epochs,
                              lr=lr,
                              eps=eps,
                              weight_decay=weight_decay,
                              num_warmup_steps=num_warmup_steps,
                              seed_val=seed_val,
                              experiments_dir=experiments_dir,
                              experiment_name=experiment_name,
                              write=write,
                              logging_steps=logging_steps)
