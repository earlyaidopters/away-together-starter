"""Second candidate: independently pretrained NLI model, locally adapted.
Upstream task training is disclosed; this is not a model trained from scratch.
"""
import json,time,threading
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification,AutoTokenizer,AutoConfig
from .model import group_logits
NLI_ID='MoritzLaurer/ModernBERT-base-zeroshot-v2.0'
NLI_REV='d421c4545a438fd006fb43f8b981c5d908faa1e1'
HYPOTHESES={
'refund':['A full cash refund is available when cancelling before the deadline.','A full cash refund is not available even when cancelling before the deadline.','It is not specified whether a full cash refund is available.'],
'arrival':['Check-in after midnight is allowed without arranging it beforehand.','Check-in after midnight is not allowed without arranging it beforehand.','Whether check-in after midnight is possible is not specified.'],
'facility':['Guests can use a swimming pool at this property for no extra charge.','Guests cannot use a swimming pool at this property for no extra charge.','It is not specified whether guests can use a swimming pool at this property for no extra charge.'],
'activity':['The price includes a hike led by a guide.','The price does not include a hike led by a guide.','It is not specified whether the price includes a hike led by a guide.']}
class NLIModel(nn.Module):
    def __init__(self,checkpoint=None):
        super().__init__()
        if checkpoint:
            folder=Path(checkpoint);config_dir=folder/'base-config' if (folder/'base-config').exists() else folder
            config=AutoConfig.from_pretrained(config_dir,local_files_only=True)
            self.base=AutoModelForSequenceClassification.from_config(config,attn_implementation='eager').float()
            self.tokenizer=AutoTokenizer.from_pretrained(folder/'tokenizer',local_files_only=True)
        else:
            self.base=AutoModelForSequenceClassification.from_pretrained(NLI_ID,revision=NLI_REV,attn_implementation='eager',dtype=torch.float32)
            self.tokenizer=AutoTokenizer.from_pretrained(NLI_ID,revision=NLI_REV)
        self.tokenizer.model_max_length=8192
    def freeze(self,top_layers=2):
        for p in self.parameters():p.requires_grad=False
        for layer in self.base.model.layers[-top_layers:]:
            for p in layer.parameters():p.requires_grad=True
        for p in self.base.head.parameters():p.requires_grad=True
        for p in self.base.classifier.parameters():p.requires_grad=True
    def encode(self,pairs,max_length=1024):
        a=[p[0] for p in pairs];b=[p[1] for p in pairs]
        result=self.tokenizer(a,b,padding=True,return_tensors='pt',truncation=False)
        if result['input_ids'].shape[1]>max_length:raise ValueError(f"Input has {result['input_ids'].shape[1]} tokens; limit {max_length}; no silent truncation")
        return result
    def forward(self,**x):
        z=self.base(**x).logits
        return z[:,0]-z[:,1]

def pairs_for(state,qs):
    pairs=[];sizes=[]
    for q in qs:
        # All engines are supplied these same hypotheses as candidate descriptions.
        candidates=q['candidates']
        # Training used standalone travel hypotheses. Keep that serialization identical.
        is_travel=any(set(candidates)==set(h) for h in HYPOTHESES.values())
        premise=state if is_travel else state+'\nDecision question: '+q['question']
        pairs.extend((premise,c) for c in candidates);sizes.append(len(candidates))
    return pairs,sizes

class NLIEngine:
    def __init__(self,trained=True):
        from safetensors.torch import load_file
        torch.set_num_threads(8);self.device='mps' if torch.backends.mps.is_available() else 'cpu';self.model=NLIModel('models/travel-nli' if trained else None);self.lock=threading.Lock()
        self.selection=json.loads(Path('models/selection.json').read_text()) if trained else {}
        if trained:self.model.load_state_dict(load_file('models/travel-nli/model.safetensors'))
        self.model.to(self.device).eval();self.temperature=self.selection.get('temperature',1);self.name='travel-nli' if trained else 'modernbert-nli-baseline'
    def predict(self,state,questions):
        with self.lock,torch.no_grad():
            if self.device=='mps':torch.mps.synchronize()
            start=time.perf_counter();pairs,sizes=pairs_for(state,questions);all_logits=[]
            for i in range(0,len(pairs),8):
                x={k:v.to(self.device) for k,v in self.model.encode(pairs[i:i+8],max_length=4096).items()};all_logits.append(self.model(**x))
            z=group_logits(torch.cat(all_logits),sizes)/self.temperature;p=z.softmax(-1).cpu().tolist()
            if self.device=='mps':torch.mps.synchronize()
            elapsed=(time.perf_counter()-start)*1000
        return {'engine':self.name,'elapsed_ms':elapsed,'model_sha256':self.selection.get('sha256'),'answers':[{'id':q['id'],'choice':max(range(len(q['candidates'])),key=lambda i:v[i]),'probabilities':v[:len(q['candidates'])]} for q,v in zip(questions,p)],'usage':{'candidate_encodings':len(pairs),'local_model_calls':(len(pairs)+7)//8}}
