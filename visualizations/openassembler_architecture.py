"""Manim scenes for the OpenAssembler architecture.

Run, when Manim is installed:
    manim -pqh openassembler_architecture.py OpenAssemblerArchitecture
    manim -pqh openassembler_architecture.py OpenAssemblerPipelineDetail

The scene labels intentionally mirror the Mermaid files in this directory.
"""

from manim import *


NAVY = "#0f172a"
BLUE = "#2563eb"
GREEN = "#4d7c0f"
ORANGE = "#c2410c"
PURPLE = "#7e22ce"
TEAL = "#0f766e"
ROSE = "#be123c"
INK = "#172033"


def panel(title, body, color, width=3.1, height=1.1, font=15):
    rect = RoundedRectangle(corner_radius=0.08, width=width, height=height)
    rect.set_stroke(color, width=2)
    rect.set_fill(color, opacity=0.10)
    head = Text(title, font_size=font, color=color, weight=BOLD)
    content = VGroup(head)
    if body:
        text = Text(body, font_size=max(10, font - 2), color=INK, line_spacing=0.85)
        content.add(text)
    content.arrange(DOWN, buff=0.08)
    content.move_to(rect.get_center())
    return VGroup(rect, content)


def arrow_between(left, right, color=INK):
    return Arrow(left.get_right(), right.get_left(), buff=0.08, stroke_width=2, color=color)


class OpenAssemblerArchitecture(Scene):
    """Top-to-bottom architecture for engineers encountering the project."""

    def construct(self):
        title = Text("OpenAssembler compiler architecture", font_size=29, color=NAVY, weight=BOLD)
        subtitle = Text("Python / CUDA C++ / PTX -> SM80 SASS + cubin -> GPU execution", font_size=15, color=INK)
        self.play(Write(title), FadeIn(subtitle, shift=DOWN * 0.15))
        self.play(title.animate.to_edge(UP, buff=0.25), subtitle.animate.next_to(title, DOWN, buff=0.08))

        inputs = VGroup(
            panel("Python", "static @kernel\nAST subset", GREEN),
            panel("CUDA C/C++", "nvcc --ptx\nsource frontend", BLUE),
            panel("PTX", "hand-written or\nfrontend output", TEAL),
            panel("PyTorch app", "Dynamo/AOTAutograd\nInductor/Triton capture", PURPLE),
        ).arrange(RIGHT, buff=0.18).scale(0.78)
        inputs.to_edge(UP, buff=1.15)
        self.play(LaggedStart(*[Create(x) for x in inputs], lag_ratio=0.12))

        frontend = panel("Frontend convergence", "to_ptx_module()\nPTX module with .entry kernels", BLUE, width=4.0, height=1.0)
        frontend.next_to(inputs, DOWN, buff=0.42)
        arrows = VGroup(*[Arrow(x.get_bottom(), frontend.get_top(), buff=0.05, stroke_width=1.4, color=BLUE) for x in inputs])
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.08), Create(frontend))

        parse = panel("PTX model", "parse_module -> params, types, labels, CFG\nvalidate scalar or exact GEMM contract", TEAL, width=4.25, height=1.05)
        parse.next_to(frontend, DOWN, buff=0.32)
        self.play(Create(parse), GrowArrow(arrow_between(frontend, parse, TEAL)))

        scalar = panel("Scalar backend", "Selector -> fragments -> DCE/CFG liveness\ncoalesce -> linear scan -> list schedule", ORANGE, width=4.0, height=1.22)
        tensor = panel("Tensor backend", "TensorSelector -> cp.async/LDGSTS -> LDSM\nHMMA -> stage-pair policy -> drains", PURPLE, width=4.0, height=1.22)
        branches = VGroup(scalar, tensor).arrange(RIGHT, buff=0.38)
        branches.next_to(parse, DOWN, buff=0.45)
        split = Arrow(parse.get_bottom(), branches.get_top(), buff=0.08, stroke_width=2, color=INK)
        split_label = Text("detect mma / ldmatrix / cp.async", font_size=11, color=INK).next_to(split, RIGHT, buff=0.08)
        self.play(GrowArrow(split), FadeIn(split_label), Create(scalar), Create(tensor))

        encode = panel("SM80 encoding", "128-bit words: opcode + operands +\nstall/yield/scoreboards/wait/reuse", TEAL, width=4.15, height=1.12)
        encode.next_to(branches, DOWN, buff=0.48)
        merge = VGroup(*[Arrow(x.get_bottom(), encode.get_top(), buff=0.06, stroke_width=1.8, color=TEAL) for x in branches])
        self.play(LaggedStart(*[GrowArrow(x) for x in merge], lag_ratio=0.1), Create(encode))

        package = panel("ptxas container splice", "ptxas builds ELF metadata; OpenAssembler\nreplaces .text.<kernel> and repairs REGCOUNT", TEAL, width=4.55, height=1.15)
        package.next_to(encode, DOWN, buff=0.32)
        self.play(Create(package), GrowArrow(arrow_between(encode, package, TEAL)))

        outputs = VGroup(
            panel("cubin", ".text + .nv.info + constants", TEAL, width=2.65, height=0.9),
            panel("SASS", "nvdisasm listing\nword round-trip", TEAL, width=2.65, height=0.9),
            panel("manifest", "source hashes, kernels,\nregisters, verification", TEAL, width=2.65, height=0.9),
        ).arrange(RIGHT, buff=0.2)
        outputs.next_to(package, DOWN, buff=0.38)
        self.play(LaggedStart(*[Create(x) for x in outputs], lag_ratio=0.1),
                  LaggedStart(*[GrowArrow(Arrow(package.get_bottom(), x.get_top(), buff=0.05, stroke_width=1.5, color=TEAL)) for x in outputs], lag_ratio=0.1))

        runtime = panel("Runtime and evidence", "SASS emulator + PTX interpreter for verify\nlocal / SSH / Modal -> CUDA driver -> A100 SM80", ROSE, width=5.1, height=1.15)
        runtime.next_to(outputs, DOWN, buff=0.42)
        self.play(Create(runtime), GrowArrow(arrow_between(outputs, runtime, ROSE)))

        note = Text("Every level preserves an explicit contract: unsupported PTX is rejected,\nverification says when it is modeled, and GPU runs compare against ptxas.", font_size=12, color=INK)
        note.next_to(runtime, DOWN, buff=0.18)
        self.play(FadeIn(note))
        self.wait(2)


class OpenAssemblerPipelineDetail(Scene):
    """Detailed backend and packaging path, with the two optimization branches."""

    def construct(self):
        title = Text("OpenAssembler: one kernel through the backend", font_size=27, color=NAVY, weight=BOLD)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title))

        stages = [
            panel("1. PTX", ".entry gemm(...)\nregisters, memory, loop latch", TEAL, width=2.5, height=1.0),
            panel("2. IR", "V/PV values\nread/write sets\nlabels + guards", ORANGE, width=2.5, height=1.0),
            panel("3. Cleanup", "DCE\nCFG liveness\ncoalesce", ORANGE, width=2.5, height=1.0),
            panel("4. Allocate", "group intervals\nlinear scan\npredicate coloring", ORANGE, width=2.5, height=1.0),
            panel("5. Schedule", "list scheduler\nlatency edges\nscoreboards", ORANGE, width=2.5, height=1.0),
            panel("6. Encode", "128-bit SM80 word\ncontrol fields\nBRA + NOP pad", TEAL, width=2.5, height=1.0),
        ]
        row = VGroup(*stages).arrange(RIGHT, buff=0.12).scale(0.83)
        row.next_to(title, DOWN, buff=0.55)
        self.play(LaggedStart(*[Create(x) for x in row], lag_ratio=0.12))
        self.play(LaggedStart(*[GrowArrow(arrow_between(a, b, ORANGE if i < 4 else TEAL)) for i, (a, b) in enumerate(zip(row[:-1], row[1:]))], lag_ratio=0.1))

        tensor_title = Text("Tensor path when mma / ldmatrix / cp.async is present", font_size=17, color=PURPLE, weight=BOLD)
        tensor_title.next_to(row, DOWN, buff=0.55)
        self.play(FadeIn(tensor_title))
        tensor = VGroup(
            panel("copy", "cp.async / LDGSTS\ncommit + wait", PURPLE, width=2.35, height=0.95),
            panel("load", "LDSM.x4/x2\nshared swizzle", PURPLE, width=2.35, height=0.95),
            panel("compute", "HMMA m16n8k16\nFP32 accumulators", PURPLE, width=2.35, height=0.95),
            panel("overlap", "stage 0 / stage 1\npeeled final pair", PURPLE, width=2.35, height=0.95),
            panel("protect", "SB0 async\nSB1..5 ordinary\nWAR + drains", PURPLE, width=2.35, height=0.95),
        ).arrange(RIGHT, buff=0.12).scale(0.86)
        tensor.next_to(tensor_title, DOWN, buff=0.25)
        self.play(LaggedStart(*[Create(x) for x in tensor], lag_ratio=0.1))
        self.play(LaggedStart(*[GrowArrow(arrow_between(a, b, PURPLE)) for a, b in zip(tensor[:-1], tensor[1:])], lag_ratio=0.1))

        verify = panel("Verification boundary", "emulator + PTX reference interpreter\npoisoned registers catch undefined reads\nGPU run compares output to ptxas", ROSE, width=6.0, height=1.0)
        verify.next_to(tensor, DOWN, buff=0.52)
        self.play(Create(verify), GrowArrow(Arrow(row.get_bottom(), verify.get_top(), buff=0.12, stroke_width=1.8, color=ROSE)))
        self.wait(2)
