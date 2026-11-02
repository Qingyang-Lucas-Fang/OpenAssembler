"""Compile and measure eight NN/transformer operations before optimization changes.

python tools/nn_study.py --dry-run
python tools/nn_study.py --backend modal --budget 180 --rounds 3 --reps 20

Uses one GPU container, shuffled jobs, deterministic inputs, explicit launch specs,
bitwise ptxas comparisons, and timestamped JSON evidence. This is a kernel study,
not an end-to-end training benchmark. No optimizer is changed by this script.
"""
from __future__ import annotations
import argparse
import dataclasses
from datetime import datetime, timezone
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from openassembler import build, runner, project


def cases():
    def b(n, init='random', dtype='f32'):
        return {'count': n, 'dtype': dtype, 'init': init}
    def spec(name, buffers, args, threads, outputs):
        return {'buffers': buffers, 'steps': [{'kernel': name, 'block': 128,
                'grid': (threads + 127)//128, 'args': args}], 'compare': outputs}
    buf = lambda name: {'buf': name}
    integer = lambda n: {'i32': n}
    n, rows, width = 16384, 128, 128
    for name in ('relu', 'gelu'):
        yield name, spec(name, {'x': b(n), 'out': b(n, 'zeros')}, [buf('x'), buf('out'), integer(n)], n, ['out'])
    for name in ('softmax', 'layernorm'):
        yield name, spec(name, {'x': b(rows*width), 'out': b(rows*width, 'zeros')},
                         [buf('x'), buf('out'), integer(rows), integer(width)], rows, ['out'])
    yield 'adam_step', spec('adam_step', {'w': b(n), 'g': b(n), 'm': b(n), 'v': b(n,[0.05]*n),
            'out': b(n,'zeros'), 'mo': b(n,'zeros'), 'vo': b(n,'zeros')},
            [buf(s) for s in ('w','g','m','v','out','mo','vo')] + [integer(n)], n, ['out','mo','vo'])
    yield 'cross_entropy', spec('cross_entropy', {'x': b(rows*width), 'labels': b(rows,[i%width for i in range(rows)],'i32'),
            'loss': b(rows,'zeros')}, [buf('x'),buf('labels'),buf('loss'),integer(rows),integer(width)], rows,['loss'])
    m, n, k = 32, 32, 64
    yield 'matmul', spec('matmul', {'a':b(m*k),'b':b(k*n),'out':b(m*n,'zeros')},
            [buf('a'),buf('b'),buf('out'),integer(m),integer(n),integer(k)],m*n,['out'])
    yield 'attention_scores', spec('attention_scores', {'q':b(m*k),'k':b(m*k),'out':b(m*m,'zeros')},
            [buf('q'),buf('k'),buf('out'),integer(m),integer(k)],m*m,['out'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', default='modal')
    ap.add_argument('--out-dir', type=Path, default=Path('output/nn-study'))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--budget', type=float, default=180)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--skip-build', action='store_true')
    args = ap.parse_args()
    if not 1 <= args.rounds <= 10 or args.reps < 0:
        ap.error('--rounds must be 1..10 and --reps nonnegative')
    if not args.skip_build:
        res = build.compile_paths(ROOT/'examples/nn_ops.py', args.out_dir, build.Options())
        if res.rc:
            return res.rc
    manifest, warnings = runner.load(args.out_dir)
    base = runner.Plan(warnings=warnings)
    specs = {}
    for name, spec in cases():
        specs[name] = spec
        runner.spec_case(spec, manifest, args.out_dir, args.reps, base, name)
    plan = runner.Plan()
    rng = random.Random(20261103)
    for round_id in range(args.rounds):
        indices = list(range(len(base.jobs)))
        rng.shuffle(indices)
        mapping = {}
        for i in indices:
            mapping[i] = len(plan.jobs)
            plan.jobs.append(base.jobs[i])
        for c in base.cases:
            plan.cases.append(dataclasses.replace(c, source=f'round-{round_id+1}', ours=mapping[c.ours], ref=mapping[c.ref]))
    options = runner.RunOptions(out_dir=args.out_dir, backend=args.backend, budget=args.budget, job_timeout=30)
    if args.dry_run:
        runner.describe(plan, args.backend, options.gpu, args.budget, None)
        return 0
    results, device = runner.execute(plan, options)
    outcomes = runner.compare(plan, results)
    rc = runner.report(outcomes, device)
    summary = []
    for c in base.cases:
        rows = [o for o in outcomes if o.case.name == c.name]
        ours = [o.t_ours for o in rows if o.t_ours is not None]
        ref = [o.t_ref for o in rows if o.t_ref is not None]
        summary.append({'kernel':c.name,'correct':all(o.status=='OK' for o in rows),
            'ours_best_median_ms':min(ours) if ours else None,'ptxas_best_median_ms':min(ref) if ref else None,
            'speedup':min(ref)/min(ours) if ours and ref and min(ours)>0 else None,
            'rounds':[{'status':o.status,'detail':o.detail,'ours_ms':o.t_ours,'ptxas_ms':o.t_ref} for o in rows]})
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report={'device':device,'timestamp':stamp,'seed':20261103,'rounds':args.rounds,'reps':args.reps,
            'source_sha256':project.sha256_file(ROOT/'examples/nn_ops.py'), 'kernels':summary,
            'specs':specs,'manifest':manifest}
    path=args.out_dir/'runs'/f'nn-study-{stamp}.json'
    project.write_atomic(path,json.dumps(report,indent=1)+'\n')
    print(f'evidence: {path}')
    return rc

if __name__=='__main__':
    raise SystemExit(main())
