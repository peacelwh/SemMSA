"""BERT text encoder (fine-tuned or frozen)."""
import torch
import torch.nn as nn
from transformers import BertModel


class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=True, pretrained='bert-base-uncased'):
        super().__init__()
        self.model = BertModel.from_pretrained(pretrained)
        self.use_finetune = use_finetune
        if not use_finetune:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, text):
        """text: (B, 3, L) = [input_ids, attention_mask, token_type_ids]."""
        input_ids, input_mask, segment_ids = text[:, 0, :].long(), text[:, 1, :].float(), text[:, 2, :].long()
        if self.use_finetune:
            return self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)[0]
        with torch.no_grad():
            return self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)[0]
