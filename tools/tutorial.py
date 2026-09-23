"""Follow-along NLI classifier workflow. Run from the repository root.
Never writes to the shipped models or final benchmark artifacts.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
ROOT=Path(__file__).resolve().parents[1]
AGENCY=ROOT/'apps/agency'
sys.path.insert(0,str(AGENCY))

def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def records(path):
    data=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids=set();out=[]
    for row in data:
        if not isinstance(row.get('id'),str) or row['id'] in ids:raise ValueError('Missing or duplicate scenario id')
        ids.add(row['id'])
        if not isinstance(row.get('state'),str) or not row['state'].strip():raise ValueError('Each scenario needs state text')
        if not row.get('questions'):raise ValueError('Each scenario needs questions')
        for q in row['questions']:
            candidates=q.get('candidates',[])
            if len(candidates)!=3 or not all(isinstance(c,str) and c.strip() for c in candidates) or len(set(candidates))!=3:raise ValueError('Supply three distinct candidate descriptions')
            if type(q.get('gold')) is not int or q['gold'] not in range(3):raise ValueError('gold must be 0, 1 or 2')
            out.append({'scenario':row['id'],'state':row['state'],**q})
    if not out:raise ValueError('Empty dataset')
    return data,out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','download','baseline','train','test','predict'])
    p.add_argument('--run',default='workshops/my-travel-model')
    p.add_argument('--data',default=str(AGENCY/'data'))
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--limit',type=int,help='Smoke test only: first N decisions per split; never a benchmark')
    p.add_argument('--input',help='JSON file with state and three candidate descriptions')
    p.add_argument('--model',choices=['base','trained'],default='trained')
    a=p.parse_args();run=Path(a.run).resolve()
    if run==ROOT or AGENCY==run or AGENCY in run.parents:raise ValueError('Use a new workshop folder outside apps/agency')
    if a.limit is not None and a.limit<1:raise ValueError('--limit must be positive')
    if a.epochs<1:raise ValueError('--epochs must be positive')
    if a.stage=='prepare':
        source=Path(a.data).resolve();loaded={s:records(source/f'{s}.jsonl') for s in ['train','dev','test']}
        for i,left in enumerate(loaded):
            for right in list(loaded)[i+1:]:
                ids={r['id'] for r in loaded[left][0]}&{r['id'] for r in loaded[right][0]}
                texts={r['state'].strip() for r in loaded[left][0]}&{r['state'].strip() for r in loaded[right][0]}
                if ids or texts:raise ValueError(f'Exact overlap between {left} and {right}; fix splits first')
        run.mkdir(parents=True,exist_ok=False);(run/'data').mkdir()
        for split in loaded:(run/'data'/f'{split}.jsonl').write_bytes((source/f'{split}.jsonl').read_bytes())
        write(run/'data-manifest.json',{s:{'sha256':digest(run/'data'/f'{s}.jsonl'),'scenarios':len(v[0]),'decisions':len(v[1])} for s,v in loaded.items()})
        print(f'Prepared {run}. Exact overlap checked; review related templates manually. No training started.');return
    from travel_lab.nli import NLIModel,NLI_ID,NLI_REV
    if a.stage=='download':
        from huggingface_hub import snapshot_download
        path=snapshot_download(repo_id=NLI_ID,revision=NLI_REV,allow_patterns=['*.json','*.safetensors','*.txt','*.model','README.md','LICENSE*'])
        print(json.dumps({'model':NLI_ID,'revision':NLI_REV,'cache_path':path}));return
    if not (run/'data-manifest.json').exists():raise ValueError('Run prepare first')
    manifest=json.loads((run/'data-manifest.json').read_text())
    for split,m in manifest.items():
        if digest(run/'data'/f'{split}.jsonl')!=m['sha256']:raise ValueError(f'{split} changed: create a new workshop for changed data')
    import torch,random
    from safetensors.torch import save_file,load_file
    from travel_lab.train_nli import evaluate,minibatch
    from travel_lab.model import group_logits
    torch.set_num_threads(8);torch.manual_seed(20260921);random.seed(20260921)
    device='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    def read(split):
        rows=records(run/'data'/f'{split}.jsonl')[1]
        return rows[:a.limit] if a.limit else rows
    if a.stage=='baseline':
        if (run/'baseline.json').exists():raise ValueError('Baseline already exists; preserve it')
        m=NLIModel().to(device);metrics,_,_=evaluate(m,read('dev'),device)
        write(run/'baseline.json',{'model':NLI_ID,'revision':NLI_REV,'split':'dev','metrics':metrics,'smoke_limit':a.limit,'decisions':len(read('dev'))});print(json.dumps(metrics));return
    if a.stage=='train':
        baseline=json.loads((run/'baseline.json').read_text())
        if baseline['smoke_limit']!=a.limit:raise ValueError('Use the same smoke/full scope as the baseline')
        modeldir=run/'checkpoint';modeldir.mkdir(exist_ok=False)
        m=NLIModel().to(device);m.freeze(2);opt=torch.optim.AdamW([x for x in m.parameters() if x.requires_grad],lr=5e-6,weight_decay=.01)
        train=read('train');dev=read('dev');best=-1;history=[];started=time.time()
        for epoch in range(a.epochs):
            m.train();random.shuffle(train);losses=[]
            for i in range(0,len(train),8):
                x,s,y=minibatch(m,train[i:i+8],device,True);opt.zero_grad();loss=torch.nn.functional.cross_entropy(group_logits(m(**x),s),y);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();losses.append(float(loss.detach()))
            metrics,_,_=evaluate(m,dev,device);row={'epoch':epoch+1,'loss':sum(losses)/len(losses),'dev':metrics,'seconds':time.time()-started};history.append(row);print(json.dumps(row),flush=True)
            if metrics['macro_f1']>best:
                best=metrics['macro_f1'];save_file({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},modeldir/'model.safetensors')
                m.tokenizer.save_pretrained(modeldir/'tokenizer');m.base.config.save_pretrained(modeldir/'base-config')
        write(run/'training.json',{'model':NLI_ID,'revision':NLI_REV,'history':history,'smoke_limit':a.limit,'device':device,'recipe':'tutorial: top two layers and head, batch 8, AdamW 5e-6; not exact historical reproduction'})
        write(run/'freeze.json',{'sha256':digest(modeldir/'model.safetensors'),'data':manifest,'smoke_limit':a.limit});print('Saved checkpoint. Demo model unchanged.');return
    freeze=json.loads((run/'freeze.json').read_text());modeldir=run/'checkpoint'
    if manifest!=freeze['data']:raise ValueError('Frozen dataset manifest changed')
    if digest(modeldir/'model.safetensors')!=freeze['sha256']:raise ValueError('Frozen checkpoint changed')
    if a.stage=='test':
        if a.limit!=freeze['smoke_limit']:raise ValueError('Use the same smoke/full scope as training')
        if (run/'test.json').exists():raise ValueError('Final test already opened; inspect its saved report instead')
        test=read('test');report={'scope':'smoke check only' if a.limit else 'held-out test','decisions':len(test),'model_sha256':freeze['sha256'],'data_sha256':manifest['test']['sha256'],'models':{}}
        for name in ['base','trained']:
            m=NLIModel(modeldir if name=='trained' else None)
            if name=='trained':m.load_state_dict(load_file(str(modeldir/'model.safetensors')))
            m.to(device);metrics,z,y=evaluate(m,test,device);pred=z.argmax(-1).tolist()
            report['models'][name]={'metrics':metrics,'rows':[{'scenario':r['scenario'],'question':r['id'],'gold':int(g),'predicted':v,'correct':v==int(g)} for r,g,v in zip(test,y.tolist(),pred)]};del m
            if device=='mps':torch.mps.empty_cache()
        write(run/'test.json',report);print(json.dumps({k:v['metrics'] for k,v in report['models'].items()}));return
    if not a.input:raise ValueError('predict requires --input')
    sample=json.loads(Path(a.input).read_text());cs=sample['candidates']
    if len(cs)!=3 or not all(isinstance(x,str) for x in cs):raise ValueError('Expected three candidate sentences')
    m=NLIModel(modeldir if a.model=='trained' else None)
    if a.model=='trained':m.load_state_dict(load_file(str(modeldir/'model.safetensors')))
    m.to(device).eval()
    with torch.no_grad():z=m(**{k:v.to(device) for k,v in m.encode([(sample['state'],c) for c in cs]).items()});probs=z.softmax(-1).cpu().tolist()
    winner=max(range(3),key=lambda i:probs[i]);print(json.dumps({'choice':winner,'answer':cs[winner],'probabilities':probs,'calibrated':False,'model':a.model}))
if __name__=='__main__':main()
