import os,json,time,random,hashlib,copy
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from safetensors.torch import save_file
from .model import CandidateModel,candidate_text,group_logits,MODEL_ID,REVISION

ROOT=Path('.')
def load(split):return [json.loads(x) for x in (ROOT/'data'/f'{split}.jsonl').read_text().splitlines()]
def flatten(rows):return [{'scenario':r['id'],'state':r['state'],**q} for r in rows for q in r['questions']]
def batch(model,items,device,permute=False):
    texts=[];sizes=[];gold=[]
    for row in items:
        order=list(range(len(row['candidates'])))
        if permute:random.shuffle(order)
        texts.extend(candidate_text(row['state'],row['question'],row['candidates'][i]) for i in order)
        sizes.append(len(order));gold.append(order.index(row['gold']))
    encoded={k:v.to(device) for k,v in model.encode(texts).items()}
    return encoded,sizes,torch.tensor(gold,device=device)
def evaluate(m,rows,device):
    m.eval();pred=[];gold=[];logits=[]
    with torch.no_grad():
        for start in range(0,len(rows),8):
            x,s,y=batch(m,rows[start:start+8],device)
            z=group_logits(m(**x),s);logits.extend(z.cpu().tolist());pred.extend(z.argmax(-1).cpu().tolist());gold.extend(y.cpu().tolist())
    return {'macro_f1':float(f1_score(gold,pred,average='macro')),'accuracy':float(np.mean(np.array(pred)==gold))},torch.tensor(logits),torch.tensor(gold)
def save(m,path,extra):
    path.mkdir(parents=True,exist_ok=True)
    save_file({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},path/'model.safetensors')
    m.tokenizer.save_pretrained(path/'tokenizer')
    (path/'config.json').write_text(json.dumps({'base':MODEL_ID,'revision':REVISION,'architecture':'masked mean pooling + 128 GELU + candidate scalar','max_length':512,**extra},indent=2))

def main():
    torch.manual_seed(20260920);random.seed(20260920);np.random.seed(20260920);torch.set_num_threads(8)
    device='mps' if torch.backends.mps.is_available() else 'cpu';m=CandidateModel().to(device)
    train=flatten(load('train'));dev=flatten(load('dev'));started=time.time();log=Path('runs/training.jsonl')
    initial=m.head[0].weight.detach().cpu().clone();history=[]
    for stage,unfreeze,epochs in [('frozen-head',0,1),('travel-v1',4,3)]:
        m.freeze(unfreeze)
        opt=torch.optim.AdamW([{'params':m.head.parameters(),'lr':1e-3},{'params':[p for p in m.encoder.parameters() if p.requires_grad],'lr':2e-5}],weight_decay=.01)
        best=-1;best_weights=None
        for epoch in range(epochs):
            m.train();random.shuffle(train);losses=[];opt.zero_grad();start=time.time()
            for i in range(0,len(train),8):
                x,s,y=batch(m,train[i:i+8],device,permute=True);z=group_logits(m(**x),s)
                loss=torch.nn.functional.cross_entropy(z,y);(loss/2).backward()
                if ((i//8)+1)%2==0 or i+8>=len(train):
                    torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();opt.zero_grad()
                losses.append(loss.item())
                if i%400==0: print(json.dumps({'stage':stage,'epoch':epoch,'questions':i,'loss':float(np.mean(losses[-50:])),'elapsed':time.time()-started}),flush=True)
                if time.time()-started>3*3600:raise TimeoutError('Training budget reached; last checkpoint retained')
            metrics,_,_=evaluate(m,dev,device)
            receipt={'stage':stage,'epoch':epoch+1,'loss':float(np.mean(losses)),'dev':metrics,'seconds':time.time()-start};history.append(receipt)
            with log.open('a') as f:f.write(json.dumps(receipt)+'\n')
            print(json.dumps(receipt),flush=True)
            if metrics['macro_f1']>best:
                best=metrics['macro_f1'];best_weights={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};save(m,Path('models')/stage,{'stage':stage,'epoch':epoch+1,'dev':metrics})
        m.load_state_dict(best_weights)
    # Choose only from development; final test remains unread.
    configs=[json.loads((Path('models')/s/'config.json').read_text()) for s in ['frozen-head','travel-v1']]
    chosen=['frozen-head','travel-v1'][int(configs[1]['dev']['macro_f1']>=configs[0]['dev']['macro_f1'])]
    from safetensors.torch import load_file
    m.load_state_dict(load_file(str(Path('models')/chosen/'model.safetensors')))
    _,logits,gold=evaluate(m,flatten(load('calibration')),device)
    temps=np.linspace(.5,4,36);losses=[torch.nn.functional.cross_entropy(logits/t,gold).item() for t in temps];temperature=float(temps[np.argmin(losses)])
    probs=(logits/temperature).softmax(-1);conf,pred=probs.max(-1)
    # Select acceptance confidence by calibration only. Need >=30 accepted and <=5% empirical errors.
    threshold=1.01
    for v in np.linspace(.5,.99,50):
        accepted=(pred==0)&(conf>=v)
        if accepted.sum()>=30 and ((gold[accepted]!=0).float().mean().item()<=.05):threshold=float(v);break
    result={'selected':chosen,'temperature':temperature,'accept_threshold':threshold,'calibration_n':len(gold),'head_weight_delta_l2':float((m.head[0].weight.detach().cpu()-initial).norm()),'training_seconds':time.time()-started,'history':history}
    weight=Path('models')/chosen/'model.safetensors';result['sha256']=hashlib.sha256(weight.read_bytes()).hexdigest()
    result['frozen_at_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
    Path('models/selection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
