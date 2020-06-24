
from transformers import BertModel, BertPreTrainedModel
from transformers import RobertaConfig, RobertaModel

from torch import nn, sigmoid
from torch.nn import MSELoss

# TODO - try ROBERTa

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


class RoBERTaMultiTaskRanker(RobertaModel):
    """ Bert Multi-Task ranking model for passage and entity ranking. """

    valid_head_flags = ['entity', 'passage']
    config = RobertaConfig()

    def __init__(self, path=None):
        super().__init__(self.config)
        # Initialise BERT setup.
        if path == None:
            self.bert = RobertaModel(self.config).from_pretrained('roberta-base')
        else:
            self.bert = RobertaModel(self.config).from_pretrained(path)
        # Dropout standard of 0.1.
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        # Head for passage ranking between 0 (not relevant) & 1 (relevant)
        self.passage_head = nn.Linear(self.config.hidden_size, 1)
        # Head for entity ranking between 0 (not relevant) & 1 (relevant)
        self.entity_head = nn.Linear(self.config.hidden_size, 1)


    def __get_BERT_outputs(self, input_ids, attention_mask):
        """ Returns BERT outputs (last_hidden_state, pooler_output, hidden_states, attentions) """
        return self.bert(input_ids=input_ids, attention_mask=attention_mask)


    def __get_BERT_cls_vector(self, input_ids, attention_mask):
        """ Returns BERT pooled_output (i.e. CLS vector) applying dropout. """
        # Get BERT outputs.
        outputs = self.__get_BERT_outputs(input_ids=input_ids, attention_mask=attention_mask)
        # Apply dropout to pooled_output (i.e. CLS vector) and apply dropout.
        pooled_output = outputs[1]
        return self.dropout(pooled_output)


    def __get_MSE(self, logits, labels):
        """ Calculate mean squared error (MSE) from logits given labels. """
        loss_fct = MSELoss()
        return loss_fct(logits.view(-1), labels.view(-1))


    def forward_head(self, head_flag='passage', input_ids=None, attention_mask=None, labels=None):
        """ Forward pass over BERT + passage head. Returns loss and logits. """
        # Get BERT CLS vector.
        cls_vector = self.__get_BERT_cls_vector(input_ids=input_ids, attention_mask=attention_mask)

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
    model = RoBERTaMultiTaskRanker()
