"""Guard the tutorial against contaminated splits and overwriting working runs."""
import importlib.util,json,subprocess,sys
from pathlib import Path
import pytest
TOOL=Path(__file__).with_name('tutorial.py')
spec=importlib.util.spec_from_file_location('tutorial',TOOL);t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
def row(id,text):return {'id':id,'state':text,'questions':[{'id':'q','candidates':['Yes.','No.','Unknown.'],'gold':0}]}
def source(tmp):
 d=tmp/'source';d.mkdir()
 for split in ['train','dev','test']:(d/f'{split}.jsonl').write_text(json.dumps(row(split,split+' independent document'))+'\n')
 return d
def prepare(data,out):return subprocess.run([sys.executable,str(TOOL),'prepare','--data',str(data),'--run',str(out)],capture_output=True,text=True)
def test_new_workspace_and_refuse_overwrite(tmp_path):
 d=source(tmp_path);out=tmp_path/'run';assert prepare(d,out).returncode==0
 before=(out/'data-manifest.json').read_bytes();assert prepare(d,out).returncode!=0;assert (out/'data-manifest.json').read_bytes()==before
 assert set(json.loads(before))=={'train','dev','test'}
def test_identical_documents_rejected_before_workspace(tmp_path):
 d=source(tmp_path);(d/'test.jsonl').write_text(json.dumps(row('other','train independent document'))+'\n');out=tmp_path/'run';r=prepare(d,out);assert r.returncode!=0;assert 'Exact overlap' in r.stderr;assert not out.exists()
@pytest.mark.parametrize('bad',[{'id':'x','state':'','questions':[]},row('x','some document')|{'questions':[{'id':'q','candidates':['a','b','c'],'gold':True}]},row('x','some document')|{'questions':[{'id':'q','candidates':['a','a','c'],'gold':0}]}])
def test_invalid_records_rejected(tmp_path,bad):
 f=tmp_path/'data.jsonl';f.write_text(json.dumps(bad));
 with pytest.raises(ValueError):t.records(f)
def test_duplicate_scenario_rejected(tmp_path):
 f=tmp_path/'data.jsonl';s=json.dumps(row('same','some document'));f.write_text(s+'\n'+s)
 with pytest.raises(ValueError):t.records(f)

def frozen_run(tmp_path):
 d=source(tmp_path);out=tmp_path/'run';assert prepare(d,out).returncode==0
 checkpoint=out/'checkpoint';checkpoint.mkdir();(checkpoint/'model.safetensors').write_bytes(b'test-only checkpoint')
 manifest=json.loads((out/'data-manifest.json').read_text())
 (out/'freeze.json').write_text(json.dumps({'sha256':t.digest(checkpoint/'model.safetensors'),'data':manifest,'smoke_limit':8}))
 return out

def invoke_test(out):
 return subprocess.run([sys.executable,str(TOOL),'test','--run',str(out),'--limit','8'],capture_output=True,text=True)

def test_final_report_cannot_be_overwritten(tmp_path):
 out=frozen_run(tmp_path);report=out/'test.json';report.write_text('{"preserve": true}')
 r=invoke_test(out);assert r.returncode!=0;assert 'Final test already opened' in r.stderr
 assert report.read_text()=='{"preserve": true}'

def test_changed_checkpoint_rejected_before_loading(tmp_path):
 out=frozen_run(tmp_path);(out/'checkpoint/model.safetensors').write_bytes(b'changed')
 r=invoke_test(out);assert r.returncode!=0;assert 'Frozen checkpoint changed' in r.stderr;assert not (out/'test.json').exists()

def test_changed_data_rejected_before_loading(tmp_path):
 out=frozen_run(tmp_path);(out/'data/test.jsonl').write_text(json.dumps(row('new','changed test document')))
 r=invoke_test(out);assert r.returncode!=0;assert 'test changed' in r.stderr;assert not (out/'test.json').exists()

def test_rewritten_manifest_rejected_after_freeze(tmp_path):
 out=frozen_run(tmp_path);data=out/'data/test.jsonl';data.write_text(json.dumps(row('new','changed test document')))
 manifest=json.loads((out/'data-manifest.json').read_text());manifest['test']['sha256']=t.digest(data)
 (out/'data-manifest.json').write_text(json.dumps(manifest))
 r=invoke_test(out);assert r.returncode!=0;assert 'Frozen dataset manifest changed' in r.stderr;assert not (out/'test.json').exists()
