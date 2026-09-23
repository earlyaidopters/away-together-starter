import json,time,random,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from safetensors.torch import save_file
from .nli import NLIModel,NLI_ID,NLI_REV,HYPOTHESES,pairs_for
from .model import group_logits
from .train import load,flatten

def rows(split):
    data=flatten(load(split))
    for r in data:r['candidates']=HYPOTHESES[r['id']]
    return data

def minibatch(m,items,device,shuffle=False):
    pairs=[];sizes=[];gold=[]
    for r in items:
        order=list(range(3))
        if shuffle:random.shuffle(order)
        pairs.extend((r['state'],r['candidates'][j]) for j in order);sizes.append(3);gold.append(order.index(r['gold']))
    return {k:v.to(device) for k,v in m.encode(pairs).items()},sizes,torch.tensor(gold,device=device)
def evaluate(m,data,device):
    m.eval();zs=[];ys=[]
    with torch.no_grad():
        for start in range(0,len(data),8):
            x,s,y=minibatch(m,data[start:start+8],device);zs.append(group_logits(m(**x),s).cpu());ys.append(y.cpu())
    z=torch.cat(zs);y=torch.cat(ys);p=z.argmax(-1)
    return {'macro_f1':float(f1_score(y,p,average='macro')),'accuracy':float((p==y).float().mean())},z,y

def main():
    torch.manual_seed(20260921);random.seed(20260921);torch.set_num_threads(8);start=time.time()
    device='mps' if torch.backends.mps.is_available() else 'cpu';m=NLIModel().to(device);dev=rows('dev');train=rows('train')
    baseline,_,_=evaluate(m,dev,device);Path('runs/nli-baseline-dev.json').write_text(json.dumps(baseline));print('baseline development',baseline,flush=True)
    initial=m.base.classifier.weight.detach().cpu().clone();m.freeze(2)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=5e-6,weight_decay=.01);best=-1;history=[]
    out=Path('models/travel-nli');out.mkdir(parents=True,exist_ok=True)
    for epoch in range(2):
        m.train();random.shuffle(train);losses=[];opt.zero_grad()
        for i in range(0,len(train),8):
            x,s,y=minibatch(m,train[i:i+8],device,True);z=group_logits(m(**x),s);loss=torch.nn.functional.cross_entropy(z,y);(loss/2).backward()
            if ((i//8)+1)%2==0 or i+8>=len(train):torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();opt.zero_grad()
            losses.append(loss.item())
            if i%800==0:print('nli',epoch,i,float(np.mean(losses[-100:])),time.time()-start,flush=True)
        metrics,_,_=evaluate(m,dev,device);receipt={'stage':'travel-nli','epoch':epoch+1,'loss':float(np.mean(losses)),'dev':metrics,'elapsed_seconds':time.time()-start};history.append(receipt);print(receipt,flush=True)
        with Path('runs/training.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
        if metrics['macro_f1']>best:
            best=metrics['macro_f1'];save_file({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},out/'model.safetensors');(out/'config.json').write_text(json.dumps({'base':NLI_ID,'revision':NLI_REV,'epoch':epoch+1,'dev':metrics,'architecture':'Pretrained binary NLI logit difference; grouped candidate softmax; top two encoder layers + original head locally fine-tuned','upstream_training':'NLI and classification mixture; overlap with public task domains, not a pure unseen-domain base'},indent=2))
    from safetensors.torch import load_file
    m.load_state_dict(load_file(str(out/'model.safetensors')));metrics,z,y=evaluate(m,rows('calibration'),device)
    temps=np.linspace(.5,8,76);temperature=float(temps[np.argmin([torch.nn.functional.cross_entropy(z/t,y).item() for t in temps])]);p=(z/temperature).softmax(-1);conf,pred=p.max(-1);threshold=1.01
    for v in np.linspace(.5,.99,50):
        accepted=(pred==0)&(conf>=v)
        if accepted.sum()>=30 and (y[accepted]!=0).float().mean().item()<=.05:threshold=float(v);break
    selection={'selected':'travel-nli','architecture':'nli','base':NLI_ID,'revision':NLI_REV,'dev':json.loads((out/'config.json').read_text())['dev'],'temperature':temperature,'accept_threshold':threshold,'calibration_n':len(y),'calibration_metrics':metrics,'head_weight_delta_l2':float((m.base.classifier.weight.detach().cpu()-initial).norm()),'training_seconds':time.time()-start,'history':history,'sha256':hashlib.sha256((out/'model.safetensors').read_bytes()).hexdigest(),'frozen_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'baseline_dev':baseline}
    m.tokenizer.save_pretrained(out/'tokenizer');m.base.config.save_pretrained(out/'base-config')
    if Path('models/selection.json').exists():shutil.copyfile('models/selection.json','models/first-candidate-selection.json')
    Path('models/selection.json').write_text(json.dumps(selection,indent=2));Path('models/hypotheses.json').write_text(json.dumps(HYPOTHESES,indent=2));print(json.dumps(selection),flush=True)
if __name__=='__main__':main()
