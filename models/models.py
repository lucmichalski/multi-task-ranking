
from transformers import BertModel, BertPreTrainedModel
from torch import nn, sigmoid
from torch.nn import MSELoss


class BertMultiTaskRanker(BertPreTrainedModel):
    """  """

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.passage_head = nn.Linear(config.hidden_size, 1)
        self.entity_head = nn.Linear(config.hidden_size, 1)
        self.init_weights()


    def __get_BERT_outputs(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds):
        """ Given BERT inputs return outputs """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
        return outputs


    def __get_BERT_pooled_outputs(self, outputs):
        """ """
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        return pooled_output


    def forward_passage(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                        head_mask=None, inputs_embeds=None, labels=None):
        """ """

        outputs = self.__get_BERT_outputs(input_ids=input_ids, attention_mask=attention_mask,
                                          token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask,
                                          inputs_embeds=inputs_embeds)
        pooled_output = self.__get_BERT_pooled_outputs(outputs=outputs)

        logits = sigmoid(self.passage_head(pooled_output))
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Add loss to outputs at index #0.
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


    def forward_entity(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                        head_mask=None, inputs_embeds=None, labels=None):
        """ """

        outputs = self.__get_BERT_outputs(input_ids=input_ids, attention_mask=attention_mask,
                                          token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask,
                                          inputs_embeds=inputs_embeds)
        pooled_output = self.__get_BERT_pooled_outputs(outputs=outputs)

        logits = sigmoid(self.passage_head(pooled_output))
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Add loss to outputs at index #0.
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
