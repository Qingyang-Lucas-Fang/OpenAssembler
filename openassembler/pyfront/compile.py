"""The actual Python-subset -> PTX compiler: ast.parse the decorated
function's source, walk the tree, emit PTX text. One Python variable name
gets exactly one fixed PTX register for its whole scope (reused across loop
iterations and if/else branches, same discipline as the hand-written PTX
elsewhere in this repo) -- no SSA, no phi nodes, by design: this is a small
enough subset that a classic one-register-per-variable allocator is both
correct and trivial to verify.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field

from openassembler.pyfront.types import ArrayType, Scalar

HEADER = ".version 8.7\n.target {arch}\n.address_size 64\n\n"

_INTRINSICS = {
    "tid_x": "%tid.x", "tid_y": "%tid.y", "tid_z": "%tid.z",
    "ctaid_x": "%ctaid.x", "ctaid_y": "%ctaid.y", "ctaid_z": "%ctaid.z",
    "ntid_x": "%ntid.x", "ntid_y": "%ntid.y", "ntid_z": "%ntid.z",
}


class NotSupported(Exception):
    """Raised for any Python construct outside the compiled subset, with the
    source line attached -- we never guess at semantics we haven't modeled."""

    def __init__(self, msg: str, node: ast.AST | None = None):
        if node is not None:
            msg = f"{msg} (line {getattr(node, 'lineno', '?')})"
        super().__init__(msg)


@dataclass
class _Reg:
    name: str      # "%r3", "%f2", "%p1", "%rd1"
    kind: str      # "s32" | "u32" | "f32" | "pred" | "ptr" (ptr = array base, b64)


@dataclass
class _Scope:
    """One function's codegen state: register allocation, labels, output."""

    arch: str
    vars: dict[str, _Reg] = field(default_factory=dict)
    arrays: dict[str, _Reg] = field(default_factory=dict)  # name -> %rd pointer reg
    params: list[tuple[str, str]] = field(default_factory=list)  # (ptx_type, name)
    lines: list[str] = field(default_factory=list)
    n_r: int = 0
    n_f: int = 0
    n_p: int = 0
    n_rd: int = 0
    label_n: int = 0
    exit_label: str = "L_exit"

    def emit(self, text: str) -> None:
        self.lines.append(text)

    def new_label(self, tag: str) -> str:
        self.label_n += 1
        return f"L_{tag}_{self.label_n}"

    def new_reg(self, kind: str) -> _Reg:
        if kind in ("s32", "u32"):
            self.n_r += 1
            return _Reg(f"%r{self.n_r}", kind)
        if kind == "f32":
            self.n_f += 1
            return _Reg(f"%f{self.n_f}", kind)
        if kind == "pred":
            self.n_p += 1
            return _Reg(f"%p{self.n_p}", kind)
        if kind == "ptr":
            self.n_rd += 1
            return _Reg(f"%rd{self.n_rd}", kind)
        raise ValueError(kind)

    def var_reg(self, name: str, kind: str, node: ast.AST) -> _Reg:
        """First assignment to `name` allocates its register; later ones must
        agree on type -- reassigning a scalar's type is not supported (it
        would need a second physical register, defeating the point of
        reusing one name across branches/iterations)."""
        if name in self.vars:
            r = self.vars[name]
            if r.kind != kind:
                raise NotSupported(f"variable {name!r} reassigned with a different type "
                                    f"({r.kind} -> {kind})", node)
            return r
        r = self.new_reg(kind)
        self.vars[name] = r
        return r


def _ann_type(ann: ast.expr) -> Scalar | ArrayType:
    """Resolve a parameter's type annotation (i32, f32, Array(f32), ...) by
    evaluating it against openassembler.pyfront.types -- annotations in this
    subset are always one of those, never arbitrary expressions."""
    from openassembler.pyfront import types as T

    src = ast.unparse(ann)
    try:
        return eval(src, {"Array": T.Array, "i32": T.i32, "u32": T.u32, "f32": T.f32})  # noqa: S307
    except Exception as e:  # noqa: BLE001
        raise NotSupported(f"unsupported parameter annotation {src!r}: {e}", ann) from e


def _param_ptx_type(t: Scalar | ArrayType) -> str:
    return "u64" if isinstance(t, ArrayType) else t.ptx


def _scalar_kind(t: Scalar) -> str:
    return t.ptx


class _Emitter(ast.NodeVisitor):
    """One pass over the function body; every visit_* either emits PTX for
    that node or raises NotSupported. Expressions return a _Reg holding the
    computed value (allocating a fresh scratch register unless the
    expression is just a bare variable read, which returns its home reg)."""

    def __init__(self, scope: _Scope):
        self.s = scope

    # ---- statements ----

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if getattr(self, "_in_kernel", False):
            raise NotSupported("nested function definitions are not supported", node)
        self._in_kernel = True
        for stmt in node.body:
            self.visit(stmt)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise NotSupported("multiple assignment targets", node)
        target = node.targets[0]
        if isinstance(target, ast.Subscript):
            self._store_subscript(target, node.value)
            return
        if not isinstance(target, ast.Name):
            raise NotSupported("assignment target must be a name or a[i]", node)
        val, kind = self._eval(node.value)
        dst = self.s.var_reg(target.id, kind, node)
        self._mov(dst, val, kind)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Subscript):
            cur, kind = self._eval(ast.Subscript(value=node.target.value, slice=node.target.slice,
                                                   ctx=ast.Load(), lineno=node.lineno))
            rhs, kind2 = self._eval(node.value)
            if kind != kind2:
                raise NotSupported("augmented assignment with mismatched types", node)
            res = self.s.new_reg(kind)
            self._binop(res, node.op, cur, rhs, kind, node)
            self._store_subscript(node.target, None, precomputed=res)
            return
        if not isinstance(node.target, ast.Name):
            raise NotSupported("augmented assignment target must be a name or a[i]", node)
        cur = self.s.vars.get(node.target.id)
        if cur is None:
            raise NotSupported(f"{node.target.id} used before assignment", node)
        rhs, kind = self._eval(node.value)
        if kind != cur.kind:
            raise NotSupported("augmented assignment with mismatched types", node)
        self._binop(cur, node.op, cur, rhs, kind, node)

    def visit_If(self, node: ast.If) -> None:
        else_label = self.s.new_label("else")
        end_label = self.s.new_label("endif")
        # `and`/`or` in the test are short-circuited directly to else_label
        # instead of being combined into one predicate register: our
        # backend's instruction selector has no case for PTX's and.pred/
        # or.pred (openassembler.backend.ra.Selector.select), only setp itself.
        self._branch_on_false(node.test, else_label)
        for stmt in node.body:
            self.visit(stmt)
        self.s.emit(f" bra {end_label};")
        self.s.emit(f"{else_label}:")
        for stmt in node.orelse:
            self.visit(stmt)
        self.s.emit(f"{end_label}:")

    def _branch_on_false(self, test: ast.expr, false_label: str) -> None:
        """Emit `@!p bra false_label` for a plain predicate, or short-circuit
        and/or without ever materializing a combined predicate register."""
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            for v in test.values:
                self._branch_on_false(v, false_label)
            return
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
            # a or b: skip straight to the body (not false_label) the moment
            # either side is true; only fall through to false_label if none are.
            true_label = self.s.new_label("or_true")
            for v in test.values[:-1]:
                self._branch_on_true(v, true_label)
            self._branch_on_false(test.values[-1], false_label)
            self.s.emit(f"{true_label}:")
            return
        pred, kind = self._eval(test)
        if kind != "pred":
            raise NotSupported("if condition must be a comparison or and/or of comparisons", test)
        self.s.emit(f" @!{pred.name} bra {false_label};")

    def _branch_on_true(self, test: ast.expr, true_label: str) -> None:
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
            for v in test.values:
                self._branch_on_true(v, true_label)
            return
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            skip_label = self.s.new_label("and_skip")
            self._branch_on_false(test, skip_label)
            self.s.emit(f" bra {true_label};")
            self.s.emit(f"{skip_label}:")
            return
        pred, kind = self._eval(test)
        if kind != "pred":
            raise NotSupported("if condition must be a comparison or and/or of comparisons", test)
        self.s.emit(f" @{pred.name} bra {true_label};")

    def visit_For(self, node: ast.For) -> None:
        # for i in range(start, stop[, step]):  -- the only loop form supported.
        if not (isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range"):
            raise NotSupported("only 'for i in range(...)' loops are supported", node)
        args = node.iter.args
        if len(args) == 1:
            start_e, stop_e, step = ast.Constant(value=0), args[0], 1
        elif len(args) == 2:
            start_e, stop_e, step = args[0], args[1], 1
        elif len(args) == 3:
            step_node = args[2]
            sign = 1
            if isinstance(step_node, ast.UnaryOp) and isinstance(step_node.op, (ast.USub, ast.UAdd)):
                sign = -1 if isinstance(step_node.op, ast.USub) else 1
                step_node = step_node.operand          # `-2` parses as UnaryOp(USub, Constant(2))
            if (not isinstance(step_node, ast.Constant) or not isinstance(step_node.value, int)
                    or isinstance(step_node.value, bool)):
                raise NotSupported("range() step must be a constant int", node)
            start_e, stop_e, step = args[0], args[1], sign * step_node.value
        else:
            raise NotSupported("range() takes 1-3 args", node)
        if step == 0:
            raise NotSupported("range() step must not be 0", node)
        if node.orelse:
            raise NotSupported("'for ... else' is not supported", node.orelse[0])
        target = node.target.id
        for sub in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            tgts = []
            if isinstance(sub, ast.Assign):
                tgts = sub.targets
            elif isinstance(sub, (ast.AugAssign, ast.AnnAssign)):
                tgts = [sub.target]
            elif isinstance(sub, ast.For):
                tgts = [sub.target]
            if any(isinstance(t_, ast.Name) and t_.id == target for t_ in tgts):
                # Python rebinds the loop variable on the next iteration; here it IS the counter,
                # so an assignment would change how many times the loop runs.
                raise NotSupported(f"assigning to the loop variable {target!r} inside its own "
                                    "loop is not supported", sub)

        i_reg = self.s.var_reg(node.target.id, "s32", node)
        start_v, start_k = self._eval(start_e)
        stop_v, stop_k = self._eval(stop_e)
        if start_k != "s32" or stop_k != "s32":
            raise NotSupported("range() bounds must be int32 expressions", node.iter)
        self._mov(i_reg, start_v, "s32")
        stop_reg = self.s.new_reg("s32")
        self._mov(stop_reg, stop_v, "s32")

        loop_label = self.s.new_label("loop")
        end_label = self.s.new_label("loopend")
        self.s.emit(f"{loop_label}:")
        pred = self.s.new_reg("pred")
        cmp = "ge" if step > 0 else "le"
        self.s.emit(f" setp.{cmp}.s32 {pred.name}, {i_reg.name}, {stop_reg.name};")
        self.s.emit(f" @{pred.name} bra {end_label};")
        for stmt in node.body:
            self.visit(stmt)
        if step == 1:
            self.s.emit(f" add.s32 {i_reg.name}, {i_reg.name}, 1;")
        elif step == -1:
            self.s.emit(f" sub.s32 {i_reg.name}, {i_reg.name}, 1;")
        else:
            op = "add" if step > 0 else "sub"
            self.s.emit(f" {op}.s32 {i_reg.name}, {i_reg.name}, {abs(step)};")
        self.s.emit(f" bra {loop_label};")
        self.s.emit(f"{end_label}:")

    def visit_Expr(self, node: ast.Expr) -> None:
        # Expression statements: only `arr[i] = ...` goes through visit_Assign;
        # a bare call statement (no side effect we model) is not supported.
        raise NotSupported("expression statement with no supported effect", node)

    def visit_Pass(self, node: ast.Pass) -> None:
        pass

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            raise NotSupported("kernels don't return a value (write to an output array)", node)
        # A bare `return` is the standard CUDA early-out (`if i >= n: return`). It used to be
        # silently dropped, so every thread went on to execute the rest of the kernel.
        self.s.emit(f" bra {self.s.exit_label};")

    def generic_visit(self, node: ast.AST) -> None:
        raise NotSupported(f"unsupported statement: {type(node).__name__}", node)

    # ---- helpers ----

    def _mov(self, dst: _Reg, src: _Reg, kind: str) -> None:
        if dst.name == src.name:
            return
        op = "mov.f32" if kind == "f32" else "mov.u32" if kind != "pred" else "mov.pred"
        self.s.emit(f" {op} {dst.name}, {src.name};")

    def _binop(self, dst: _Reg, op: ast.operator, a: _Reg, b: _Reg, kind: str, node: ast.AST) -> None:
        signed = kind == "s32"
        table = {
            ast.Add: ("add.f32" if kind == "f32" else "add.s32"),
            ast.Sub: ("sub.f32" if kind == "f32" else "sub.s32"),
            ast.Mult: ("mul.f32" if kind == "f32" else "mul.lo.s32"),
            ast.Div: ("div.rn.f32" if kind == "f32" else None),
            ast.FloorDiv: (None if kind == "f32" else f"div.{'s32' if signed else 'u32'}"),
            ast.Mod: (None if kind == "f32" else f"rem.{'s32' if signed else 'u32'}"),
        }
        mnem = table.get(type(op))
        if mnem is None:
            raise NotSupported(f"operator {type(op).__name__} not supported for {kind}", node)
        self.s.emit(f" {mnem} {dst.name}, {a.name}, {b.name};")

    def _store_subscript(self, target: ast.Subscript, value_node: ast.expr | None,
                          precomputed: _Reg | None = None) -> None:
        if not isinstance(target.value, ast.Name) or target.value.id not in self.s.arrays:
            raise NotSupported("subscript target must be a declared array", target)
        base = self.s.arrays[target.value.id]
        idx, idx_kind = self._eval(target.slice)
        if idx_kind != "s32":
            raise NotSupported(f"array index must be an int32 expression, not {idx_kind}", target)
        elem_kind = self._array_elem_kind(target.value.id)
        if precomputed is not None:
            val, val_kind = precomputed, elem_kind
        else:
            val, val_kind = self._eval(value_node)
        if val_kind != elem_kind:
            # `out[i] = 1` into an Array(f32) used to store the integer's raw bits
            # (1.4e-45, not 1.0) with no error -- and --verify can't see it, because the
            # interpreter and the SASS agree on the wrong value.
            raise NotSupported(f"storing a {val_kind} value into an Array of {elem_kind} "
                                "(write 1.0, not 1, for a float array)", target)
        addr = self.s.new_reg("ptr")
        # mul.wide.u32 (not .s32): our backend fuses mul.wide+add.s64 of a
        # cbank-held pointer into one IMAD.WIDE, and the encoder only has a
        # template for the unsigned form of that fusion (R,R,R,CBank with no
        # .U32 modifier isn't templated -- see openassembler/isa/encode.py).
        # Array indices are non-negative in practice, so this costs nothing.
        self.s.emit(f" mul.wide.u32 {addr.name}, {idx.name}, {4};")
        self.s.emit(f" add.s64 {addr.name}, {base.name}, {addr.name};")
        store_ty = "f32" if elem_kind == "f32" else "u32"
        self.s.emit(f" st.global.{store_ty} [{addr.name}], {val.name};")

    def _array_elem_kind(self, name: str) -> str:
        return self._elem_kinds[name]

    _elem_kinds: dict[str, str] = {}

    def _eval(self, node: ast.expr, want_pred: bool = False) -> tuple[_Reg, str]:
        if isinstance(node, ast.Constant):
            return self._const(node)
        if isinstance(node, ast.Name):
            if node.id not in self.s.vars:
                raise NotSupported(f"{node.id!r} used before assignment", node)
            r = self.s.vars[node.id]
            return r, r.kind
        if isinstance(node, ast.BinOp):
            a, ka = self._eval(node.left)
            b, kb = self._eval(node.right)
            if ka != kb:
                raise NotSupported(f"mixed-type operands ({ka} vs {kb}); cast explicitly", node)
            dst = self.s.new_reg(ka)
            self._binop(dst, node.op, a, b, ka, node)
            return dst, ka
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.BoolOp):
            return self._boolop(node)
        if isinstance(node, ast.UnaryOp):
            return self._unaryop(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, ast.Subscript):
            return self._load_subscript(node)
        raise NotSupported(f"unsupported expression: {type(node).__name__}", node)

    def _const(self, node: ast.Constant) -> tuple[_Reg, str]:
        v = node.value
        if isinstance(v, bool):
            raise NotSupported("bool constants are not supported directly", node)
        if isinstance(v, int):
            r = self.s.new_reg("s32")
            self.s.emit(f" mov.s32 {r.name}, {v};")
            return r, "s32"
        if isinstance(v, float):
            import struct

            bits = struct.unpack("<I", struct.pack("<f", v))[0]
            r = self.s.new_reg("f32")
            self.s.emit(f" mov.f32 {r.name}, 0f{bits:08X};")
            return r, "f32"
        raise NotSupported(f"unsupported constant type: {type(v).__name__}", node)

    def _compare(self, node: ast.Compare) -> tuple[_Reg, str]:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotSupported("chained comparisons (a < b < c) are not supported", node)
        a, ka = self._eval(node.left)
        b, kb = self._eval(node.comparators[0])
        if ka != kb:
            raise NotSupported(f"comparing mismatched types ({ka} vs {kb})", node)
        op_map = {ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge", ast.Eq: "eq", ast.NotEq: "ne"}
        cmp = op_map.get(type(node.ops[0]))
        if cmp is None:
            raise NotSupported(f"comparison {type(node.ops[0]).__name__} not supported", node)
        dst = self.s.new_reg("pred")
        self.s.emit(f" setp.{cmp}.{ka} {dst.name}, {a.name}, {b.name};")
        return dst, "pred"

    def _boolop(self, node: ast.BoolOp) -> tuple[_Reg, str]:
        # `and`/`or` as the direct test of `if`/`while` are short-circuited by
        # _branch_on_false/_branch_on_true (visit_If) without ever reaching
        # here. Anywhere else -- assigned to a variable, nested in another
        # expression -- would need a combined predicate register, which our
        # backend's instruction selector has no case for (and.pred/or.pred).
        raise NotSupported("'and'/'or' are only supported directly in an if condition "
                            "(e.g. 'if a < b and c < d:'), not assigned to a variable or nested "
                            "in another expression", node)

    def _unaryop(self, node: ast.UnaryOp) -> tuple[_Reg, str]:
        v, k = self._eval(node.operand)
        if isinstance(node.op, ast.USub):
            dst = self.s.new_reg(k)
            mnem = "neg.f32" if k == "f32" else "neg.s32"
            self.s.emit(f" {mnem} {dst.name}, {v.name};")
            return dst, k
        if isinstance(node.op, ast.Not) and k == "pred":
            dst = self.s.new_reg("pred")
            self.s.emit(f" not.pred {dst.name}, {v.name};")
            return dst, "pred"
        raise NotSupported(f"unary operator {type(node.op).__name__} not supported for {k}", node)

    def _call(self, node: ast.Call) -> tuple[_Reg, str]:
        if not isinstance(node.func, ast.Name):
            raise NotSupported("only calls to a bare name are supported", node)
        name = node.func.id
        if name in _INTRINSICS:
            if node.args:
                raise NotSupported(f"{name}() takes no arguments", node)
            r = self.s.new_reg("s32")
            self.s.emit(f" mov.u32 {r.name}, {_INTRINSICS[name]};")
            return r, "s32"
        if name in ("float", "int"):
            if len(node.args) != 1:
                raise NotSupported(f"{name}() takes exactly one argument", node)
            v, k = self._eval(node.args[0])
            want = "f32" if name == "float" else "s32"
            if k == want:
                return v, k
            dst = self.s.new_reg(want)
            self.s.emit(f" cvt.rn.{want}.{k} {dst.name}, {v.name};" if want == "f32"
                        else f" cvt.rzi.{want}.{k} {dst.name}, {v.name};")
            return dst, want
        raise NotSupported(f"call to {name!r} is not supported", node)

    def _load_subscript(self, node: ast.Subscript) -> tuple[_Reg, str]:
        if not isinstance(node.value, ast.Name) or node.value.id not in self.s.arrays:
            raise NotSupported("subscript target must be a declared array", node)
        base = self.s.arrays[node.value.id]
        idx, idx_kind = self._eval(node.slice)
        if idx_kind != "s32":
            raise NotSupported("array index must be an int32 expression", node)
        elem_kind = self._array_elem_kind(node.value.id)
        addr = self.s.new_reg("ptr")
        self.s.emit(f" mul.wide.u32 {addr.name}, {idx.name}, 4;")  # see note in _store_subscript
        self.s.emit(f" add.s64 {addr.name}, {base.name}, {addr.name};")
        load_ty = "f32" if elem_kind == "f32" else "u32"
        dst = self.s.new_reg(elem_kind)
        self.s.emit(f" ld.global.{load_ty} {dst.name}, [{addr.name}];")
        return dst, elem_kind


def _get_source(fn) -> ast.FunctionDef:
    lines, first = inspect.getsourcelines(fn)
    src = textwrap.dedent("".join(lines))
    mod = ast.parse(src)
    ast.increment_lineno(mod, first - 1)     # report real file line numbers, not "line 2"
    (fdef,) = [n for n in mod.body if isinstance(n, ast.FunctionDef)]
    return fdef


def compile_kernel(fn, arch: str = "sm_80") -> str:
    """Compile a @kernel-decorated function's source to PTX text. Raises
    NotSupported for anything outside the subset in this module's docstring."""
    real_fn = fn.__wrapped__ if hasattr(fn, "__wrapped__") else fn
    fdef = _get_source(real_fn)
    sig = inspect.signature(real_fn)

    scope = _Scope(arch=arch)
    elem_kinds: dict[str, str] = {}
    for pname, param in sig.parameters.items():
        if param.annotation is inspect.Parameter.empty:
            raise NotSupported(f"parameter {pname!r} has no type annotation", fdef)
        t = _ann_type(_annotation_node(fdef, pname))
        ptx_t = _param_ptx_type(t)
        scope.params.append((ptx_t, pname))
        if isinstance(t, ArrayType):
            r = scope.new_reg("ptr")
            scope.arrays[pname] = r
            elem_kinds[pname] = t.elem.ptx
        else:
            r = scope.new_reg(_scalar_kind(t))
            scope.vars[pname] = r

    emitter = _Emitter(scope)
    emitter._elem_kinds = elem_kinds
    emitter.visit(fdef)

    return _render(real_fn.__name__, scope)


def _annotation_node(fdef: ast.FunctionDef, pname: str) -> ast.expr:
    for a in fdef.args.args:
        if a.arg == pname:
            if a.annotation is None:
                raise NotSupported(f"parameter {pname!r} has no type annotation", fdef)
            return a.annotation
    raise NotSupported(f"parameter {pname!r} not found", fdef)


def _render(name: str, s: _Scope) -> str:
    param_list = ", ".join(f".param .{t} {n}" for t, n in s.params)
    out = [HEADER.format(arch=s.arch), f".visible .entry {name}({param_list})\n{{"]
    decls = []
    if s.n_p:
        decls.append(f" .reg .pred %p<{s.n_p + 1}>;")
    if s.n_r:
        decls.append(f" .reg .b32 %r<{s.n_r + 1}>;")
    if s.n_f:
        decls.append(f" .reg .f32 %f<{s.n_f + 1}>;")
    if s.n_rd or s.arrays:
        decls.append(f" .reg .b64 %rd<{s.n_rd + 1}>;")
    out.extend(decls)
    for ptx_t, pname in s.params:
        if pname in s.arrays:
            r = s.arrays[pname]
            out.append(f" ld.param.u64 {r.name}, [{pname}];")
            out.append(f" cvta.to.global.u64 {r.name}, {r.name};")
        else:
            r = s.vars[pname]
            ld = "f32" if ptx_t == "f32" else "u32" if ptx_t in ("s32", "u32") else ptx_t
            out.append(f" ld.param.{ld} {r.name}, [{pname}];")
    out.extend(s.lines)
    out.append(f"{s.exit_label}:")
    out.append(" ret;\n}")
    return "\n".join(out) + "\n"


def kernel(fn):
    """Marks a function as pyfront-compilable. Calling it directly in plain
    Python raises -- use openassembler.pyfront.compile_kernel(fn) to get PTX."""

    def _no_call(*_a, **_kw):
        raise NotSupported(f"{fn.__name__} is a @kernel function; call "
                            f"openassembler.pyfront.compile_kernel({fn.__name__}) to get PTX, "
                            "not the function itself")

    _no_call.__wrapped__ = fn
    _no_call.__name__ = fn.__name__
    _no_call._is_openassembler_kernel = True
    return _no_call


def find_kernel(path, entry: str | None):
    """Import a .py file as a standalone module and return the @kernel
    function named `entry`, or the only one if there's exactly one and entry
    is None. Raises NotSupported if none or several match ambiguously --
    same "don't guess" policy as the rest of this subset compiler."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernels = {name: obj for name, obj in vars(mod).items()
               if callable(obj) and getattr(obj, "_is_openassembler_kernel", False)}
    if entry:
        if entry not in kernels:
            raise NotSupported(f"no @kernel function named {entry!r} in {path}; have {list(kernels)}")
        return kernels[entry]
    if len(kernels) != 1:
        raise NotSupported(f"{path} has {len(kernels)} @kernel functions, pass --kernel NAME; "
                            f"have {list(kernels)}")
    (fn,) = kernels.values()
    return fn
