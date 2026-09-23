"""Candidate-wise ModernBERT classifier. No hosted model required at inference."""
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

MODEL_ID = 'answerdotai/ModernBERT-base'
REVISION = '8949b909ec900327062f0ebf497f51aef5e6f0c8'

class CandidateModel(nn.Module):
    def __init__(self, model_id=MODEL_ID, revision=REVISION):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id, revision=revision, attn_implementation='eager', dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.head = nn.Sequential(nn.Linear(self.encoder.config.hidden_size,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1))
    def freeze(self, top_layers=0):
        for p in self.encoder.parameters(): p.requires_grad = False
        if top_layers:
            for layer in self.encoder.layers[-top_layers:]:
                for p in layer.parameters(): p.requires_grad = True
    def forward(self,input_ids,attention_mask):
        h=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        mask=attention_mask.unsqueeze(-1).to(h.dtype)
        pooled=(h*mask).sum(1)/mask.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)
    def encode(self, texts, max_length=512):
        lengths=self.tokenizer(texts, truncation=False, add_special_tokens=True)['input_ids']
        if max(map(len,lengths))>max_length: raise ValueError(f'Input exceeds {max_length} tokens; refusing silent truncation')
        return self.tokenizer(texts,padding=True,truncation=False,return_tensors='pt')

def candidate_text(state, question, candidate):
    return f'Document: {state}\nRequirement: {question}\nCandidate decision: {candidate}'

def group_logits(flat, sizes):
    chunks=flat.split(sizes)
    return nn.utils.rnn.pad_sequence(chunks,batch_first=True,padding_value=float('-inf'))
