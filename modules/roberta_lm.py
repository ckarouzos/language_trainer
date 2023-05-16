import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel


class RobertaForLM(RobertaPreTrainedModel):
    """Roberta Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings)"""
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        model_outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = model_outputs[0]
        logits = self.lm_head(hidden_states)
        outputs = (logits,) + model_outputs[1:]
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)