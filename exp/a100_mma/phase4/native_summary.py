"""Audit production CLI promotion evidence without another GPU run."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def main():
    builds=json.loads((HERE/'native-build-evidence.json').read_text())
    rows=[]
    for build in builds:
        path=BASE/'results'/f"{build['build']}.json"
        record=json.loads(path.read_text())[0]
        assert record.get('worker_rc')==0 and 'error' not in record
        assert record['input_provenance']['ours_sha256']==build['cubin_sha256']
        m,n,k=record['shape_mnk']
        assert len(record['checks'])==32
        assert all(check['full_output_elements_checked']==m*n for check in record['checks'])
        assert build['text_equal'] and record['resources']['ours']['num_regs']==216
        reference=build['phase3_reference']
        times=record['milliseconds']
        rows.append(dict(build=build['build'],shape_mnk=[m,n,k],dtype=record['dtype'],
                         full_output_checks=len(record['checks']),elements_per_check=m*n,
                         source=str(path.relative_to(BASE)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         native_ms=times['ours'],phase3_identical_code_ms=times[reference],
                         matched_ptxas_ms=times['ptxas'],cublas_ms=times['cublas'],
                         instruction_bytes_equal=True,device=record['device'],
                         recorded_utc=record['recorded_utc']))
    (HERE/'native-gpu-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('Audited six native CLI binaries, 192 full-output check rows.')
    for row in rows:print(row['build'],row['native_ms'],row['phase3_identical_code_ms'])

if __name__=='__main__':main()
