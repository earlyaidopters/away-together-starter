"""Convert checked support examples into explicit candidate-scoring records; does not train."""
import argparse,json
from pathlib import Path
LABELS=['billing','technical_support','needs_review']
def convert(rows,split):
    seen=set();out=[]
    for row in rows:
        text=row.get('text','').strip();label=row.get('label');id=row.get('id')
        if not isinstance(id,str) or not id or id in seen:raise ValueError('Each example needs a unique nonempty id.')
        if len(text)<8 or label not in LABELS:raise ValueError('A complete text and a supported checked label are required.')
        seen.add(id);out.append({'id':id,'split':split,'state':text,'questions':[{'id':'route','question':'Which team should review this support request?','candidates':['The request concerns billing or payment.','The request concerns a technical problem.','The request needs manual review before routing.'],'labels':LABELS,'gold':LABELS.index(label)}],'source':'viewer-labeled example; no quality validation implied'})
    if not out:raise ValueError('No examples supplied.')
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--split',required=True,choices=['train','dev','test']);p.add_argument('--out',required=True);a=p.parse_args()
    result=convert([json.loads(l) for l in Path(a.input).read_text().splitlines() if l.strip()],a.split)
    with Path(a.out).open('x') as f:
        for r in result:f.write(json.dumps(r)+'\n')
    print(f'Prepared {len(result)} {a.split} records. Configure a new task and trainer, keep test cases separate, and evaluate before use. No model was trained.')
