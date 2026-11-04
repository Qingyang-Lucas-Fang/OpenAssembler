"""All candidates in the same warmed worker, on identical inputs and clocks."""
import json
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
def config(name,wn=64):
    return dict(build=name,name=name,m=4096,n=4096,k=4096,dtype='f16',tile_m=128,tile_n=128,
                threads=256 if wn==32 else 128,smem=65536)
root=config('validated_square_f16');root['name']='p2_paired_windows'
root['steady_warmup_rounds']=16
root['alternatives']=[config('p2_window0_w64x32',32),config('p2_window1_w64x32',32),
                      config('p2_window2_w64x32',32),config('p2_window4_w64x64')]
(BASE/'phase2/paired-suite.json').write_text(json.dumps([root],indent=2)+'\n')
