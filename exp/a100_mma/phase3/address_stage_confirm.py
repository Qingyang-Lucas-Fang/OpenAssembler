"""Confirm constant stage-pair specialization across six matched projections.

Reuses all six row+B-pair kernels and the two existing square stage-pair
kernels. Compiles only four new rectangular kernels through the isolated
address_shared/address_stage helpers. No GPU execution is performed.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(HERE))
import address_shared as address


def verify_config(config,*,stage_pair):
    directory=BASE/'build'/config['build']
    metadata=json.loads((directory/'experiment.json').read_text())
    geometry=metadata['geometry']
    for key in ('m','n','k'):
        assert config[key]==geometry[key],(config['build'],key)
    manifest=json.loads((directory/'manifest.json').read_text())
    entry=manifest['entries'][0]
    source=(directory/entry['ptx']).read_bytes()
    expected=metadata.get('source_ptx_sha256',metadata.get('source_sha256'))
    assert hashlib.sha256(source).hexdigest()==expected
    candidate=metadata['candidate']
    if stage_pair:
        assert candidate['stage_pair'] and candidate['shared_chunk_immediates']
        assert candidate['paired_b'] and candidate['row_epilogue'] and candidate['gap']==8
        assert metadata['spliced_words_and_controls_exact']
        assert metadata['exact_nvdisasm_tensor_memory_store_operands']
    else:
        assert candidate['paired_b'] and candidate['row'] and candidate['gap']==8
        assert candidate['role_bias']==0 and not candidate.get('paired_preaddress')
    assert config['dtype'] in ('f16','bf16') and config['smem']==65536 and config['threads']==128
    return dict(build=config['build'],ptx_sha256=expected,
                cubin_sha256=hashlib.sha256((directory/entry['cubin']).read_bytes()).hexdigest(),
                geometry=geometry,candidate=candidate,
                scheduled_instructions=metadata['kernels'][0]['sass_instructions'],
                registers=metadata['kernels'][0]['registers'])


def main():
    address.search.backend.install()
    reference_path=HERE/'search_combined_suite.json'
    references=json.loads(reference_path.read_text())
    suite=[];provenance=[]
    for reference in references:
        dtype=reference['dtype'];m,n,k=(reference[key] for key in ('m','n','k'))
        shape=next(name for name,dims in [('square',(4096,4096,4096)),
                   ('mlp',(4096,16384,4096)),('tall',(16384,4096,4096))] if dims==(m,n,k))
        matches=[config for config in reference['alternatives']
                 if config['build']==f'search_combined_{shape}_row_bpair_{dtype}']
        if len(matches)!=1:raise ValueError('expected one unmodified row+B-pair reference')
        baseline=dict(matches[0]);baseline.pop('alternatives',None)
        if shape=='square':
            candidate=dict(baseline,build=f'address_stagepair_square_{dtype}',
                           name=f'address_stagepair_square_{dtype}')
        else:
            candidate=address.compile_case(m,n,k,dtype,tag=f'confirm_{shape}',stage_pair=True)
        baseline_info=verify_config(baseline,stage_pair=False)
        candidate_info=verify_config(candidate,stage_pair=True)
        baseline['name']=f'address_stage_confirm_{shape}_{dtype}'
        baseline['steady_warmup_rounds']=16
        baseline['alternatives']=[candidate]
        suite.append(baseline)
        provenance.append(dict(shape=shape,dtype=dtype,baseline=baseline_info,candidate=candidate_info,
                               candidate_reused=shape=='square'))
    assert len(suite)==6
    (HERE/'address-stage-confirm-suite.json').write_text(json.dumps(suite,indent=2)+'\n')
    (HERE/'address-stage-confirm-provenance.json').write_text(json.dumps(dict(
          input_suite_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
          builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          cases=provenance),indent=2)+'\n')
    print(json.dumps(dict(cases=len(suite),kernels=12,new_rectangular_builds=4,
                          reused_square_stagepair_builds=2),indent=2))


if __name__=='__main__':main()
