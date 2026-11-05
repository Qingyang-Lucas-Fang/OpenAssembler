"""Manim scenes for OpenAssembler's concrete speedup mechanisms.

Run, when Manim is installed:
    manim -pqh openassembler_speedups.py SpeedupOverview
    manim -pqh openassembler_speedups.py ConstantDivision
    manim -pqh openassembler_speedups.py LoopUnrollAndSchedule
    manim -pqh openassembler_speedups.py RegisterAllocation
    manim -pqh openassembler_speedups.py TensorStagePair
    manim -pqh openassembler_speedups.py AlphaTensorFastMM
"""

from manim import *


NAVY = "#0f172a"
ORANGE = "#c2410c"
TEAL = "#0f766e"
PURPLE = "#7e22ce"
BLUE = "#2563eb"
GREEN = "#4d7c0f"
ROSE = "#be123c"


def card(title, body, color, width=3.0, height=1.0, font=15):
    r = RoundedRectangle(corner_radius=0.08, width=width, height=height)
    r.set_stroke(color, width=2)
    r.set_fill(color, opacity=0.10)
    t = Text(title, font_size=font, color=color, weight=BOLD)
    g = VGroup(t)
    if body:
        b = Text(body, font_size=max(10, font - 2), color=NAVY, line_spacing=0.82)
        g.add(b)
    g.arrange(DOWN, buff=0.06).move_to(r.get_center())
    return VGroup(r, g)


def link(a, b, color=NAVY):
    return Arrow(a.get_right(), b.get_left(), buff=0.07, stroke_width=2, color=color)


class SpeedupOverview(Scene):
    def construct(self):
        title = Text("OpenAssembler speedup map", font_size=29, color=NAVY, weight=BOLD).to_edge(UP, buff=0.25)
        self.play(Write(title))
        items = [
            card("divconst", "magic multiply-high\n13 -> 2-3 instructions", TEAL),
            card("unroll", "counted loop x4\nhide predicate latency", ORANGE),
            card("IR + RA", "DCE, liveness, coalesce\nlinear-scan register groups", ORANGE),
            card("schedule", "latency graph ->\nstall / SB / wait / reuse", ORANGE),
            card("tensor", "LDSM + HMMA\nasync stage-pair overlap", PURPLE),
            card("fastmm", "AlphaTensor / Strassen\ncompile-time fused tiles", BLUE),
        ]
        grid = VGroup(*items).arrange_in_grid(rows=2, cols=3, buff=(0.22, 0.35)).scale(0.86)
        grid.next_to(title, DOWN, buff=0.55)
        self.play(LaggedStart(*[Create(x) for x in grid], lag_ratio=0.1))
        footer = Text("These are component-kernel mechanisms; measured gains are workload- and schedule-specific.", font_size=13, color=ROSE)
        footer.next_to(grid, DOWN, buff=0.35)
        self.play(FadeIn(footer))
        self.wait(2)


class ConstantDivision(Scene):
    def construct(self):
        title = Text("A. Constant division and remainder lowering", font_size=25, color=NAVY, weight=BOLD).to_edge(UP, buff=0.25)
        self.play(Write(title))
        src = card("PTX input", "div.u32 q, x, 37\nrem.u32 r, x, 37", BLUE, width=2.8, height=1.05)
        opt = card("opt.divconst", "classify d\npow2 or magic\nquotient sequence", TEAL, width=2.8, height=1.05)
        proof = card("SMT gate", "Z3 proves all 2^32 inputs\nproved / refuted / unknown", GREEN, width=2.8, height=1.05)
        sass = card("SM80 SASS", "IMAD.HI magic\nSHF / correction\nIMAD remainder", ORANGE, width=3.1, height=1.05)
        row = VGroup(src, opt, proof, sass).arrange(RIGHT, buff=0.25).scale(0.9)
        row.next_to(title, DOWN, buff=0.75)
        self.play(LaggedStart(*[Create(x) for x in row], lag_ratio=0.12))
        self.play(LaggedStart(*[GrowArrow(link(a, b, TEAL)) for a, b in zip(row[:-1], row[1:])], lag_ratio=0.12))
        old = VGroup(*[Text(x, font_size=17, color=ROSE) for x in ["runtime reciprocal", "MUFU", "shift", "fixup", "...", "13 instructions"]]).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
        new = VGroup(*[Text(x, font_size=17, color=GREEN) for x in ["magic multiply-high", "shift", "optional add", "remainder x - q*d", "2-3 instructions"]]).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
        compare = VGroup(card("before", "", ROSE, width=2.1, height=2.15), card("after", "", GREEN, width=2.1, height=2.15))
        old.move_to(compare[0].get_center())
        new.move_to(compare[1].get_center())
        compare[0].add(old)
        compare[1].add(new)
        compare.next_to(row, DOWN, buff=0.75)
        self.play(Create(compare[0]), Create(compare[1]))
        result = Text("The optimizer is local, explicit, and proof-gated; unsupported cases remain visible.", font_size=14, color=NAVY)
        result.next_to(compare, DOWN, buff=0.25)
        self.play(FadeIn(result))
        self.wait(2)


class LoopUnrollAndSchedule(Scene):
    def construct(self):
        title = Text("B. Counted-loop unrolling hides predicate latency", font_size=25, color=NAVY, weight=BOLD).to_edge(UP, buff=0.25)
        self.play(Write(title))
        source = card("original loop", "body;\nadd counter;\nsetp.lt; @p bra loop", BLUE, width=2.75, height=1.15)
        unrolled = card("unroll_loops(4)", "body0 body1 body2 body3\none latch and tail", ORANGE, width=2.75, height=1.15)
        sched = card("list scheduler", "compute next predicate\nearly; fill independent slots", TEAL, width=2.95, height=1.15)
        row = VGroup(source, unrolled, sched).arrange(RIGHT, buff=0.32).scale(0.95)
        row.next_to(title, DOWN, buff=0.7)
        self.play(LaggedStart(*[Create(x) for x in row], lag_ratio=0.12))
        self.play(LaggedStart(*[GrowArrow(link(a, b, ORANGE)) for a, b in zip(row[:-1], row[1:])], lag_ratio=0.12))

        labels = ["trip 0", "trip 1", "trip 2", "trip 3", "next latch"]
        blocks = VGroup(*[card(x, "ALU / LDG / FFMA", ORANGE, width=1.35, height=0.72, font=12) for x in labels]).arrange(RIGHT, buff=0.12)
        blocks.next_to(row, DOWN, buff=0.8)
        self.play(LaggedStart(*[Create(x) for x in blocks], lag_ratio=0.08))
        pred = Text("predicate for the next trip is computed before the current body needs it", font_size=14, color=TEAL)
        pred.next_to(blocks, DOWN, buff=0.22)
        self.play(FadeIn(pred))
        latency = VGroup(
            Text("13-cycle predicate -> guard gap", font_size=16, color=ROSE),
            Text("becomes schedulable work between independent instructions", font_size=16, color=GREEN),
        ).arrange(DOWN, buff=0.1)
        latency.next_to(pred, DOWN, buff=0.35)
        self.play(FadeIn(latency))
        self.wait(2)


class RegisterAllocation(Scene):
    def construct(self):
        title = Text("C. SSA cleanup and register pressure control", font_size=25, color=NAVY, weight=BOLD).to_edge(UP, buff=0.25)
        self.play(Write(title))
        stages = VGroup(
            card("virtual IR", "v0 v1 v2 v3\np0 p1 guards", BLUE, width=2.0, height=0.95),
            card("DCE + liveness", "remove dead ops\nkeep loop-carried homes", TEAL, width=2.1, height=0.95),
            card("coalesce", "delete compatible\nwide MOV copies", GREEN, width=2.1, height=0.95),
            card("linear scan", "aligned groups\nP0..P6 coloring", ORANGE, width=2.1, height=0.95),
        ).arrange(RIGHT, buff=0.18).scale(0.95)
        stages.next_to(title, DOWN, buff=0.7)
        self.play(LaggedStart(*[Create(x) for x in stages], lag_ratio=0.1))
        self.play(LaggedStart(*[GrowArrow(link(a, b, ORANGE)) for a, b in zip(stages[:-1], stages[1:])], lag_ratio=0.1))

        axis = Line(LEFT * 5.4, RIGHT * 5.4, color=NAVY, stroke_width=2).shift(DOWN * 1.05)
        self.play(Create(axis))
        ticks = VGroup()
        for i in range(9):
            x = axis.get_start()[0] + i * (axis.get_length() / 8)
            ticks.add(Line([x, axis.get_y() - 0.08, 0], [x, axis.get_y() + 0.08, 0], color=NAVY))
            ticks.add(Text(str(i), font_size=11, color=NAVY).next_to(ticks[-1], DOWN, buff=0.04))
        self.play(Create(ticks))
        intervals = [(0, 3, BLUE, "v0"), (1, 6, ORANGE, "v1"), (2, 4, TEAL, "v2"), (5, 8, PURPLE, "v3")]
        bars = VGroup()
        for row, (a, b, color, name) in enumerate(intervals):
            y = axis.get_y() + 0.35 + row * 0.28
            start = axis.get_start()[0] + a * axis.get_length() / 8
            end = axis.get_start()[0] + b * axis.get_length() / 8
            seg = Line([start, y, 0], [end, y, 0], color=color, stroke_width=8)
            lab = Text(name, font_size=12, color=color).next_to(seg, LEFT, buff=0.08)
            bars.add(seg, lab)
        self.play(LaggedStart(*[Create(x) for x in bars], lag_ratio=0.08))
        caption = Text("allocation is interval coloring with copy affinity, not a blind one-register-per-PTX-name map", font_size=14, color=NAVY)
        caption.to_edge(DOWN, buff=0.25)
        self.play(FadeIn(caption))
        self.wait(2)


class TensorStagePair(Scene):
    def construct(self):
        title = Text("E/F. Native SM80 tensor stage-pair pipeline", font_size=25, color=NAVY, weight=BOLD).to_edge(UP, buff=0.25)
        self.play(Write(title))
        rows = [
            ("stage 0", ["cp.async", "wait_group", "LDSM", "HMMA", "STG"]),
            ("stage 1", ["cp.async", "wait_group", "LDSM", "HMMA", "STG"]),
            ("stage 0", ["cp.async", "wait_group", "LDSM", "HMMA", "STG"]),
        ]
        left = Text("time ->", font_size=15, color=NAVY).to_edge(LEFT, buff=0.3).shift(DOWN * 0.4)
        self.play(FadeIn(left))
        timeline = VGroup()
        colors = [TEAL, BLUE, PURPLE, ORANGE, ROSE]
        for r, (name, ops) in enumerate(rows):
            label = Text(name, font_size=14, color=PURPLE).to_edge(LEFT, buff=0.45).shift(DOWN * (0.2 + r * 0.85))
            timeline.add(label)
            for i, op in enumerate(ops):
                c = card(op, "", colors[i], width=1.45, height=0.58, font=12)
                c.move_to([-2.65 + i * 1.45, -0.2 - r * 0.85, 0])
                timeline.add(c)
        self.play(LaggedStart(*[Create(x) for x in timeline], lag_ratio=0.05))

        sb0 = card("SB0", "async copy completion", TEAL, width=2.2, height=0.7, font=13)
        sb1 = card("SB1..5", "ordinary memory / result\ncompletion tracking", ORANGE, width=2.2, height=0.7, font=13)
        war = card("dependency guards", "address WAR + reader tracking\nepoch drain before observers", ROSE, width=2.8, height=0.7, font=13)
        notes = VGroup(sb0, sb1, war).arrange(RIGHT, buff=0.22)
        notes.next_to(timeline, DOWN, buff=0.35)
        self.play(LaggedStart(*[Create(x) for x in notes], lag_ratio=0.1))
        footer = Text("overlap copy, load, compute, and store without recycling a scoreboard or address too early", font_size=14, color=NAVY)
        footer.next_to(notes, DOWN, buff=0.2)
        self.play(FadeIn(footer))
        self.wait(2)


class AlphaTensorFastMM(Scene):
    def construct(self):
        title = Text("G. AlphaTensor / Strassen fast matrix multiply", font_size=25, color=NAVY, weight=BOLD).to_edge(UP, buff=0.25)
        self.play(Write(title))
        scheme = card("rank-R scheme", "u, v, w factorization\n97 imported candidates", BLUE, width=2.7, height=1.0)
        exact = card("exact verifier", "rational tensor identity\nall output coefficients", GREEN, width=2.7, height=1.0)
        fuse = card("compile-time fusion", "linear combinations of A/B\ncoefficient terms bounded by registers", TEAL, width=2.9, height=1.0)
        tile = card("GPU plan", "sgemm128 tiles\nblock recursion + accumulation", PURPLE, width=2.7, height=1.0)
        row = VGroup(scheme, exact, fuse, tile).arrange(RIGHT, buff=0.24).scale(0.9)
        row.next_to(title, DOWN, buff=0.72)
        self.play(LaggedStart(*[Create(x) for x in row], lag_ratio=0.1))
        self.play(LaggedStart(*[GrowArrow(link(a, b, BLUE)) for a, b in zip(row[:-1], row[1:])], lag_ratio=0.1))

        left = Square(side_length=1.8, color=BLUE).shift(LEFT * 3.0 + DOWN * 1.35)
        right = Square(side_length=1.8, color=PURPLE).shift(RIGHT * 3.0 + DOWN * 1.35)
        for i in range(1, 4):
            self.add(Line(left.get_left() + RIGHT * i * 0.45, left.get_left() + RIGHT * i * 0.45 + UP * 1.8, color=BLUE, stroke_width=1))
            self.add(Line(left.get_bottom() + UP * i * 0.45, left.get_bottom() + UP * i * 0.45 + RIGHT * 1.8, color=BLUE, stroke_width=1))
            self.add(Line(right.get_left() + RIGHT * i * 0.45, right.get_left() + RIGHT * i * 0.45 + UP * 1.8, color=PURPLE, stroke_width=1))
            self.add(Line(right.get_bottom() + UP * i * 0.45, right.get_bottom() + UP * i * 0.45 + RIGHT * 1.8, color=PURPLE, stroke_width=1))
        self.play(Create(left), Create(right))
        self.play(GrowArrow(Arrow(left.get_right(), right.get_left(), buff=0.2, color=TEAL, stroke_width=3)))
        label = Text("rank products replace some direct block products; fused lincomb kernels feed the same tiled GEMM backend", font_size=13, color=NAVY)
        label.next_to(right, DOWN, buff=0.3)
        self.play(FadeIn(label))
        self.wait(2)
