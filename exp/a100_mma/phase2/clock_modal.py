"""Driver-only clock probe execution on a bounded A100 container."""
import base64,json,statistics,sys
from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent
app=modal.App('openassembler-a100-clocked-probes')
image=modal.Image.debian_slim(python_version='3.11').add_local_python_source('openassembler',copy=False)

@app.function(image=image,gpu='A100-40GB',timeout=180,max_containers=1,retries=0)
def execute(cases):
    from openassembler.gpu.cuda_driver import Driver,run_probe_job
    import struct
    drv=Driver();drv.open();results=[]
    for case in cases:
        try:
            job={'kname':'gemm','cubin':base64.b64decode(case['binary']),
                 'trials':[([(0,0,0,0)]*32,0,0)]*16}
            raw=run_probe_job(drv,job)
            checksum=(case.get('expected_raw') if case.get('expected_raw') is not None else
                      struct.unpack('<I',struct.pack('<f',float(case['steps']*16)))[0])
            actual=[lane[1] for trial in raw['out'] for lane in trial]
            if any(x!=checksum for x in actual):raise AssertionError(
                f'checksum failure: expected {checksum:#x}; observed {[hex(x) for x in actual[:16]]}; attrs={raw["attrs"]}')
            cycles=[trial[0][0] for trial in raw['out']]
            record={k:v for k,v in case.items() if k!='binary'}|{
                'device':vars(drv.info()),'clock_delta_cycles':cycles,
                'median_cycles':statistics.median(cycles),'checksum_ok':True,'oob_bytes':raw['oob_bytes'],
                'scope':'single warp, one CTA; clock reads include fixed drain and clock overhead; LDSM includes two dependent ALU ops'}
            if case['family'] in ('native-sparse-capability','numerical-capability'):
                record.pop('clock_delta_cycles');record.pop('median_cycles')
                record['scope']=('uniform compressed-A/B sparse-MMA capability probe; no speed or general sparse-kernel correctness claim'
                    if case['family']=='native-sparse-capability' else 'single paper-motivated truncation discriminant; no general numerical model proof')
                if any(lane[0]!=checksum for trial in raw['out'] for lane in trial):raise AssertionError('sparse output0 mismatch')
            results.append(record)
        except Exception as exc:
            results.append({'name':case['name'],'error':str(exc)})
    drv.close();return results

@app.local_entrypoint()
def main(suite_path:str='clocked/suite.json',output:str='clocked/results.json'):
    base=HERE.parent;suite=json.loads((HERE/suite_path).read_text())
    for case in suite:case['binary']=base64.b64encode((base/case['cubin_path']).read_bytes()).decode()
    result=execute.remote(suite)
    (HERE/output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k in ('name','median_cycles','error')} for r in result],indent=2))
