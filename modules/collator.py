import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DataCollatorForMaskedLanguageModeling():
    """Data collator used for masked language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling by:
        - replacing [MASK] tokens in the input with random words
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm_prob: float=0.15):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob
    
    def __call__(self, examples):

        if not isinstance(examples[0], (dict, BatchEncoding)):
            examples = [vars(f) for f in examples]
        first = examples[0]

        batch = {}
        for k, v in first.items():
            if v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([example[k] for example in examples])
                else:
                    batch[k] = torch.tensor([example[k] for example in examples])
        
        inputs, labels = self.mask_tokens(batch["input_ids"])
        return {
            "input_ids": inputs,
            "attention_mask": batch["attention_mask"],
            "labels": labels
        }
        
    def mask_tokens(self, input: torch.Tensor):

        if self.tokenizer.mask_token is None:
            raise ValueError("This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer.")
        
        labels = input.clone
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input[indices_random] = random_words[indices_random]

        return input, labels
