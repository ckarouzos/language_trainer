import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel

@dataclass
class RobertaForMultiTaskingOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits_mlm: torch.FloatTensor = None
    logits_seq_cls: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class RobertaForMultiTasking(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.seq_cls_head = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits_mlm = self.mlm_head(sequence_output)
        logits_seq_cls = self.seq_cls_head(sequence_output[:, 0, :])  

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            mlm_labels, seq_cls_labels = labels
            loss_mlm = loss_fct(logits_mlm.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            loss_seq_cls = loss_fct(logits_seq_cls.view(-1, self.config.num_labels), seq_cls_labels.view(-1))
            loss = loss_mlm + loss_seq_cls
        
        if not return_dict:
            output = (logits_mlm, logits_seq_cls) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return RobertaForMultiTaskingOutput(
            loss=loss,
            logits_mlm=logits_mlm,
            logits_seq_cls=logits_seq_cls,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )