"""Compile-time factor plans and exact verification, with explicit precision gate.

No device coefficient loads, no dynamic factor branches, and no speed claim.
Products are consumed one at a time; lowering still needs a resource cost model.
"""
import json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE.parents[1]))
from openassembler.fastmm.factors import all_schemes,verify


def plan(scheme):
    transforms={};products=[]
    for r in range(scheme.rank):
        values={}
        for name,matrix in (('A',scheme.u),('B',scheme.v)):
            expression=tuple((i,str(row[r])) for i,row in enumerate(matrix) if row[r])
            key=(name,expression)
            if key not in transforms:transforms[key]=f'{name}{len(transforms)}'
            values[name]=transforms[key]
        outputs=[{'block':c,'coefficient':str(row[r])} for c,row in enumerate(scheme.w) if row[r]]
        products.append({'product':r,'A':values['A'],'B':values['B'],'consume_immediately_into':outputs})
    return {'name':scheme.name,'block_shape':[scheme.n,scheme.m,scheme.p],
            'rank':scheme.rank,'classical_block_products':scheme.naive(),
            'multiplication_only_speed_limit':scheme.naive()/scheme.rank,
            'direct_evaluation_additions':scheme.additions(),
            'factors_are_compile_time_constants':True,'dynamic_coefficient_loads':0,
            'precision_policy':'FP16/BF16 inputs; FP32 transforms then one quantization before MMA; FP32 accumulation/output; not algebraically identical to direct floating-point GEMM',
            'transform_nodes':[{'id':name,'operand':key[0],'expression':list(key[1])} for key,name in transforms.items()],
            'product_schedule':products,'max_live_product_tiles':1,
            'implementation_gate':'No GPU low-rank speed claim until direct schedule and transform/reconstruction overhead are measured under matched precision'}


if __name__=='__main__':
    schemes=all_schemes();records=[]
    for name in ('strassen_2x2x2','at_2x2x2','at_gpu_4x4x4'):
        scheme=schemes[name]
        if not verify(scheme):raise AssertionError(name)
        records.append(plan(scheme))
    (BASE/'phase2/alphatensor-constant-plans.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in p.items() if k in ('name','rank','direct_evaluation_additions','multiplication_only_speed_limit')} for p in records],indent=2))
