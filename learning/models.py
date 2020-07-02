
from transformers import BertModel, BertPreTrainedModel

from torch import nn, sigmoid
from torch.nn import MSELoss


class BertMultiTaskRanker(BertPreTrainedModel):
    """ Bert Multi-Task ranking model for passage and entity ranking. """

    valid_head_flags = ['entity', 'passage']

    def __init__(self, config):
        super().__init__(config)
        # Initialise BERT setup.
        self.bert = BertModel(config)
        # Dropout standard of 0.1.
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Head for passage ranking between 0 (not relevant) & 1 (relevant)
        self.passage_head = nn.Linear(config.hidden_size, 1)
        # Head for entity ranking between 0 (not relevant) & 1 (relevant)
        self.entity_head = nn.Linear(config.hidden_size, 1)
        # Initialise BERT weights.
        self.init_weights()


    def __get_BERT_outputs(self, input_ids, attention_mask, token_type_ids):
        """ Returns BERT outputs (last_hidden_state, pooler_output, hidden_states, attentions) """
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)


    def __get_BERT_cls_vector(self, input_ids, attention_mask, token_type_ids):
        """ Returns BERT pooled_output (i.e. CLS vector) applying dropout. """
        # Get BERT outputs.
        outputs = self.__get_BERT_outputs(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Apply dropout to pooled_output (i.e. CLS vector) and apply dropout.
        pooled_output = outputs[1]
        return self.dropout(pooled_output)


    def __get_MSE(self, logits, labels):
        """ Calculate mean squared error (MSE) from logits given labels. """
        loss_fct = MSELoss()
        return loss_fct(logits.view(-1), labels.view(-1))


    def forward_head(self, head_flag='passage', input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        """ Forward pass over BERT + passage head. Returns loss and logits. """
        # Get BERT CLS vector.
        cls_vector = self.__get_BERT_cls_vector(input_ids=input_ids, attention_mask=attention_mask,
                                                token_type_ids=token_type_ids)

        assert head_flag in self.valid_head_flags, "head_flag: {}, valid_head_flags: {}".format(head_flag, self.valid_head_flags)
        # Calculate logits.
        if head_flag == 'passage':
            logits = sigmoid(self.passage_head(cls_vector))
        elif head_flag == 'entity':
            logits = sigmoid(self.entity_head(cls_vector))
        else:
            "NOT VALID HEAD SELECTION"
            raise

        # Calculate loss.
        loss = self.__get_MSE(logits=logits, labels=labels)

        return loss, logits


if __name__ == '__main__':
    from learning.experiments import FineTuningReRankingExperiments

    train_data_dir_path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/roberta_data/'
    train_batch_size = 2
    dev_batch_size = 128
    dev_data_dir_path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/roberta_data/'
    dev_qrels_path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/dev_benchmark_Y1_25.qrels'
    dev_run_path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/dev_benchmark_Y1_25.run'
    model_path = None #'/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/model/'
    use_token_type_ids = False
    experiment = FineTuningReRankingExperiments(model_path=model_path,
                                                train_data_dir_path=train_data_dir_path,
                                                train_batch_size=train_batch_size,
                                                dev_data_dir_path=dev_data_dir_path,
                                                dev_batch_size=dev_batch_size,
                                                dev_qrels_path=dev_qrels_path,
                                                dev_run_path=dev_run_path)

    epochs = 1
    lr = 1e-5
    eps = 1e-8
    weight_decay = 0.01
    warmup_percentage = 0.1
    experiments_dir = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/exp/'
    experiment_name = 'roberta_benchmarkY1_lr_5e5_v3'
    write = True
    logging_steps = 100
    head_flag = 'passage'

    experiment.run_experiment_single_head(
        head_flag=head_flag,
        epochs=epochs,
        lr=lr,
        eps=eps,
        weight_decay=weight_decay,
        warmup_percentage=warmup_percentage,
        experiments_dir=experiments_dir,
        experiment_name=experiment_name,
        logging_steps=logging_steps
    )

    # head_flag = 'passage'
    # rerank_run_path = '/nfs/trec_car/data/entity_ranking/test_runs/roberta_passage_testY1_1000.run'
    # experiment.inference(head_flag=head_flag, rerank_run_path=rerank_run_path, do_eval=False)

