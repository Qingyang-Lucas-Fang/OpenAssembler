"""Experimental native SM80 warp-per-row reductions, before production promotion.

The full-warp .sync.bfly form is selected only for all-32-lane participation.
The delta uses the corpus-proven register operand, and SHFL is scoreboarded
like the recorded NVIDIA sequence. No new encoding bits are guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct
import os


def enable_experimental_shuffle():
    from openassembler.backend import ra
    from openassembler.ptx.parse import ImmI, Var
    original = ra.Selector.select
    if getattr(original, "_reduction_shuffle", False):
        return

    def select(self, ins):
        if ins.op != "shfl":
            return original(self, ins)
        a = ins.args
        if (ins.mods != ("sync", "bfly", "b32") or len(a) != 5
                or not isinstance(a[0], Var)
                or not isinstance(a[3], ImmI) or a[3].value != 31
                or not isinstance(a[4], ImmI) or a[4].value != 0xffffffff):
            raise ra.NotSupported("only full-warp shfl.sync.bfly.b32 clamp=31 mask=0xffffffff")
        dest = self.new()
        if os.environ.get("OA_REDUCTION_VARIANT", "").startswith("integrated"):
            if ins.guard is not None:
                raise ra.NotSupported("predicated shfl.sync is not supported")
            mask=self.reg32(ImmI(0xffffffff))
            self.emit("WARPSYNC {s0}", [], [mask], side=True)
        self.emit("SHFL.BFLY PT, {d0}, {s0}, {s1}, 0x1f", [dest],
                  [self.reg32(a[1]), self.reg32(a[2])], var_lat=True)
        self.bind(a[0].name, dest)
    select._reduction_shuffle = True
    ra.Selector.select = select
    original_bind = ra.Selector.bind

    def bind(self, name, value):
        # A previously bound PTX value is not a fresh producer result. Retargeting
        # its last producer steals the value from its existing PTX home.
        if (name in getattr(self, "homes", {}) and isinstance(value, ra.V)
                and (value in ra._flat(self.env.values()) or value in ra._flat(self.homes.values()))):
            home = self.homes[name]
            if home is None:
                home = self.homes[name] = self.new()
            if value != home:
                self.emit("MOV {d0}, {s0}", [home], [value], copy=True)
            self.env[name] = home
            return
        return original_bind(self,name,value)
    ra.Selector.bind = bind
    variant = os.environ.get("OA_REDUCTION_VARIANT", "bind")
    if "drain" in variant:
        from dataclasses import replace
        from openassembler.backend import compile as backend_compile
        old_schedule = backend_compile.schedule
        def schedule(*args, **kwargs):
            result = old_schedule(*args, **kwargs)
            active = set()
            fixed = []
            for text, ctl in result:
                active.difference_update(b for b in range(6) if ctl.wait & (1<<b))
                newly_assigned = {b for b in (ctl.wbar,ctl.rbar) if b!=7}
                reused = active & newly_assigned
                extra = sum(1<<b for b in reused)
                fixed.append((text,replace(ctl,wait=ctl.wait|extra)))
                active.difference_update(reused)
                active.update(newly_assigned)
            return fixed
        backend_compile.schedule = schedule
    if "mufu_read" in variant:
        from openassembler.backend import compile as backend_compile
        backend_compile.MEM_ASYNC = tuple(backend_compile.MEM_ASYNC) + ("MUFU",)
    if "shuffle_read" in variant:
        from openassembler.backend import compile as backend_compile
        backend_compile.MEM_ASYNC = tuple(backend_compile.MEM_ASYNC) + ("SHFL",)
    if variant.startswith("stall"):
        from dataclasses import replace
        from openassembler.backend import compile as backend_compile
        old_schedule = backend_compile.schedule
        cycles = int(variant[5:])
        def schedule(*args, **kwargs):
            return [(text,replace(ctl,stall=max(cycles,ctl.stall),
                                  yield_=ctl.yield_ or max(cycles,ctl.stall)>11))
                    for text,ctl in old_schedule(*args,**kwargs)]
        backend_compile.schedule = schedule
    if variant == "mufu_resultwar":
        from dataclasses import replace
        from openassembler.backend import compile as backend_compile
        old_schedule = backend_compile.schedule
        def schedule(code,*args,**kwargs):
            result=old_schedule(code,*args,**kwargs)
            pending={};fixed=[]
            for op,(text,ctl) in zip(code,result):
                wait=ctl.wait
                for reg in op.writes:
                    if reg in pending:wait |= 1<<pending[reg]
                pending={r:b for r,b in pending.items() if not (wait&(1<<b))}
                fixed.append((text,replace(ctl,wait=wait)))
                if backend_compile._opname(text)=="MUFU" and ctl.wbar!=7:
                    for reg in op.reads:pending[reg]=ctl.wbar
            return fixed
        backend_compile.schedule=schedule
    if variant.startswith("integrated"):
        from openassembler.backend import compile as bc
        from openassembler.backend import sched as bs
        from openassembler.isa.word import Control
        from exp.weaknesses.reductions_warpsync_template import install
        install()
        bs.ORDERED=tuple(bs.ORDERED)+("WARPSYNC","SHFL")
        bc.MEM_ASYNC=tuple(bc.MEM_ASYNC)+("MUFU","SHFL")
        def schedule(code,labels=None,model="safe"):
            result=[];pending_w={};pending_r={};next_sb=0
            joins=set((labels or {}).values())
            def take():
                nonlocal next_sb
                b=next_sb;next_sb=(next_sb+1)%6;return b
            for i,op in enumerate(code):
                wait=0
                all_live=set(pending_w.values())|{b for bars in pending_r.values() for b in bars}
                if i in joins or bc._opname(op.text) in ("BRA","EXIT","WARPSYNC"):
                    wait=sum(1<<b for b in all_live)
                pred_reads=[("p",p) for p in bc._puses(op.text)]
                pred_writes=[("p",p) for p in bc._pdefs(op.text)]
                for r in op.reads+op.writes+pred_reads+pred_writes:
                    if r in pending_w:wait |= 1<<pending_w[r]
                for r in op.writes:
                    for b in pending_r.get(r,()):wait |= 1<<b
                wbar=take() if op.var_lat else 7
                rbar=take() if bc._opname(op.text) in bc.MEM_ASYNC and op.reads else 7
                for b in (wbar,rbar):
                    if b in all_live:wait |= 1<<b
                pending_w={r:b for r,b in pending_w.items() if not (wait&(1<<b))}
                pending_r={r:bars-{b for b in range(6) if wait&(1<<b)}
                           for r,bars in pending_r.items()}
                pending_r={r:bars for r,bars in pending_r.items() if bars}
                if wbar!=7:
                    for r in op.writes+pred_writes:pending_w[r]=wbar
                if rbar!=7:
                    for r in op.reads:pending_r.setdefault(r,set()).add(rbar)
                stall=2 if op.var_lat else 6
                if bc._opname(op.text) in ("STG","EXIT","BRA","WARPSYNC"):stall=5
                result.append((op.text,Control(stall=stall,yield_=False,wbar=wbar,rbar=rbar,wait=wait,reuse=0)))
            if model=="mined":result=bc._mined_stalls(code,result,joins)
            return bc._branch_pred_latency(result)
        bc.schedule=schedule


def _fp(x):
    return "0f" + struct.pack(">f", x).hex()


@dataclass(frozen=True, slots=True)
class ReductionPlan:
    rows: int
    width: int
    op: str
    epsilon: float = 1e-5
    arch: str = "sm_80"

    def __post_init__(self):
        if type(self.rows) is not int or self.rows <= 0 or self.rows >= 2**32:
            raise ValueError("rows must be a positive integer within CUDA grid-x")
        if type(self.width) is not int or not 1 <= self.width <= 1024:
            raise ValueError("width must be an integer in [1, 1024]")
        if self.rows * self.width > 2**32:
            raise ValueError("element indices must fit unsigned 32 bits")
        if self.op not in ("softmax", "layernorm"):
            raise ValueError("op must be softmax or layernorm")
        if self.arch != "sm_80":
            raise ValueError("parallel row reductions require sm_80")
        if (isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (int, float))):
            raise ValueError("epsilon must be positive and finite")
        try:
            epsilon_f32 = struct.unpack(">f", struct.pack(">f", self.epsilon))[0]
        except (OverflowError, struct.error):
            raise ValueError("epsilon must remain positive and finite in FP32") from None
        if not math.isfinite(epsilon_f32) or epsilon_f32 < 2**-126:
            raise ValueError("epsilon must remain positive, normal and finite in FP32 for rsqrt.ftz")

    @property
    def launch(self):
        return {"grid": [(self.rows+3)//4, 1, 1], "threads": 128, "smem": 0}

    @property
    def metadata(self):
        return {"kernel": self.op, "rows": self.rows, "width": self.width,
                "op": self.op, "epsilon": self.epsilon, "arch": self.arch,
                "input_dtype": "f32", "output_dtype": "f32", "accumulation": "f32",
                "layout": "contiguous_row_major", "warps_per_cta": 4,
                "lanes_per_row": 32, "finite_inputs_only": True,
                "numerical_contract": "bounded finite FP32 inputs; FP32 differences, sums and squares must remain finite",
                "affine": False, "input_alignment": 4, "output_alignment": 4}


def generate(plan):
    if not isinstance(plan, ReductionPlan):
        raise TypeError("generate requires a ReductionPlan")
    chunks = (plan.width+31)//32
    lines = [".version 8.7\n.target sm_80\n.address_size 64",
             f".visible .entry {plan.op}(.param .u64 X,.param .u64 Y) {{",
             ".reg .b32 %r<16>; .reg .b64 %d<8>; .reg .f32 %f<160>; .reg .pred %p<4>;",
             "ld.param.u64 %d0,[X]; ld.param.u64 %d1,[Y];",
             "mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x;",
             "and.b32 %r2,%r0,31; shr.u32 %r3,%r0,5;",
             "mad.lo.u32 %r4,%r1,4,%r3;",
             f"setp.ge.u32 %p0,%r4,{plan.rows}; @%p0 bra done;",
             f"mad.lo.u32 %r5,%r4,{plan.width},%r2;",
             "mul.wide.u32 %d2,%r5,4; add.u64 %d3,%d0,%d2; add.u64 %d4,%d1,%d2;"]
    if plan.op == "layernorm":
        lines.extend([f"mul.lo.u32 %r6,%r4,{plan.width};",
                      "mul.wide.u32 %d5,%r6,4; add.u64 %d6,%d0,%d5;",
                      "ld.global.f32 %f119,[%d6];"])
    tail = plan.width % 32
    if tail:
        lines += [f"setp.lt.u32 %p1,%r2,{tail};"]
    for j in range(chunks):
        if tail and j == chunks-1:
            neutral = "0fff800000" if plan.op == "softmax" else "0f00000000"
            lines += [f"mov.f32 %f{j},{neutral};", f"@%p1 ld.global.f32 %f{j},[%d3+{128*j}];"]
        else:
            lines += [f"ld.global.f32 %f{j},[%d3+{128*j}];"]

    def reduce(kind, src, dest):
        lines.append(f"mov.f32 %f{dest},%f{src[0]};")
        for reg in src[1:]:
            lines.append(f"{kind}.f32 %f{dest},%f{dest},%f{reg};")
        for delta in (16,8,4,2,1):
            lines.extend([f"shfl.sync.bfly.b32 %f159,%f{dest},{delta},31,0xffffffff;",
                          f"{kind}.f32 %f{dest},%f{dest},%f159;"])

    if plan.op == "softmax":
        reduce("max", list(range(chunks)), 120)
        for j in range(chunks):
            lines.extend([f"sub.f32 %f{j},%f{j},%f120;",
                          f"mul.f32 %f{j},%f{j},{_fp(math.log2(math.e))};",
                          f"ex2.approx.ftz.f32 %f{j},%f{j};"])
        reduce("add", list(range(chunks)), 121)
        lines.append("rcp.approx.ftz.f32 %f122,%f121;")
        for j in range(chunks):
            lines.append(f"mul.f32 %f{j},%f{j},%f122;")
    else:
        # Shift before summing: small variance around a large row mean must
        # not amplify FP32 sum cancellation through the normalization scale.
        for j in range(chunks):
            lines.append(f"sub.f32 %f{j},%f{j},%f119;")
            if tail and j == chunks-1:
                lines.append(f"@!%p1 mov.f32 %f{j},0f00000000;")
        reduce("add", list(range(chunks)), 120)
        lines.append(f"mul.f32 %f120,%f120,{_fp(1/plan.width)};")
        for j in range(chunks):
            lines.append(f"sub.f32 %f{j},%f{j},%f120;")
            if tail and j == chunks-1:
                # Inactive columns contribute exactly zero to population variance.
                lines.append(f"@!%p1 mov.f32 %f{j},0f00000000;")
            lines.append(f"mul.f32 %f{32+j},%f{j},%f{j};")
        reduce("add", list(range(32,32+chunks)), 121)
        lines.extend([f"mul.f32 %f121,%f121,{_fp(1/plan.width)};",
                      f"add.f32 %f121,%f121,{_fp(plan.epsilon)};",
                      "rsqrt.approx.ftz.f32 %f122,%f121;"])
        for j in range(chunks):
            lines.append(f"mul.f32 %f{j},%f{j},%f122;")
    for j in range(chunks):
        guard = "@%p1 " if tail and j == chunks-1 else ""
        lines.append(f"{guard}st.global.f32 [%d4+{128*j}],%f{j};")
    lines.extend(["done:", "ret;", "}"])
    return "\n".join(lines)+"\n"
