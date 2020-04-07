
from transformers import BertModel, BertPreTrainedModel
from torch import nn, sigmoid
from torch.nn import MSELoss


class BertMultiTaskRanker(BertPreTrainedModel):
    """ Bert Multi-Task ranking model for passage and entity ranking. """

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


    def __get_BERT_outputs(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds):
        """ Returns BERT outputs (last_hidden_state, pooler_output, hidden_states, attentions) """
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                         position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)


    def __get_BERT_cls_vector(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                              inputs_embeds):
        """ Returns BERT pooled_output (i.e. CLS vector) applying dropout. """
        # Get BERT outputs.
        outputs = self.__get_BERT_outputs(input_ids=input_ids, attention_mask=attention_mask,
                                          token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask,
                                          inputs_embeds=inputs_embeds)
        # Apply dropout to pooled_output (i.e. CLS vector) and apply dropout.
        pooled_output = outputs[1]
        return self.dropout(pooled_output)


    def __get_MSE(self, logits, labels):
        """ Calculate mean squared error (MSE) from logits given labels. """
        loss_fct = MSELoss()
        return loss_fct(logits.view(-1), labels.view(-1))


    def forward_passage(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                        head_mask=None, inputs_embeds=None, labels=None):
        """ Forward pass over BERT + passage head. Returns loss and logits. """
        # Get BERT CLS vector.
        cls_vector = self.__get_BERT_cls_vector(input_ids=input_ids, attention_mask=attention_mask,
                                                token_type_ids=token_type_ids, position_ids=position_ids,
                                                head_mask=head_mask, inputs_embeds=inputs_embeds)
        # Calculate logits.
        logits = sigmoid(self.passage_head.forward(input=cls_vector))
        # Calculate loss.
        loss = self.__get_MSE(logits=logits, labels=labels)

        return loss, logits


    def forward_entity(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                        head_mask=None, inputs_embeds=None, labels=None):
        """ Forward pass over BERT + entity head. Returns loss and logits. """
        # Get BERT CLS vector.
        cls_vector = self.__get_BERT_cls_vector(input_ids=input_ids, attention_mask=attention_mask,
                                                token_type_ids=token_type_ids, position_ids=position_ids,
                                                head_mask=head_mask, inputs_embeds=inputs_embeds)
        # Calculate logits.
        logits = sigmoid(self.entity_head.forward(input=cls_vector))
        # Calculate loss.
        loss = self.__get_MSE(logits=logits, labels=labels)

        return loss, logits
