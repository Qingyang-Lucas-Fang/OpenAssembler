"""Instrument project forward methods without modifying application source.

Only project-owned files are transformed. Library implementations and host
control flow remain Python. An injected decorator selects the actual compiler.
"""
from __future__ import annotations

import ast
import importlib.abc
import importlib.machinery
from pathlib import Path

DECORATOR = '__openassembler_compile_forward__'


def transform(source: str, filename: str) -> tuple[object, list[str]]:
    tree = ast.parse(source, filename=filename)
    targets = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for method in node.body:
            if isinstance(method, ast.FunctionDef) and method.name == 'forward':
                # Outermost: preserve existing method decorators underneath.
                method.decorator_list.insert(0, ast.Name(id=DECORATOR, ctx=ast.Load()))
                targets.append(node.name + '.forward')
    ast.fix_missing_locations(tree)
    return compile(tree, filename, 'exec'), targets


class ProjectLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, name, path, decorate):
        super().__init__(name, path)
        self.decorate = decorate

    def exec_module(self, module):
        code, _ = transform(self.get_source(module.__name__), self.path)
        module.__dict__[DECORATOR] = self.decorate
        exec(code, module.__dict__)


class ProjectFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root: Path, decorate):
        self.root, self.decorate = root.resolve(), decorate

    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if (spec is not None and spec.origin and spec.origin.endswith('.py')
                and Path(spec.origin).resolve().is_relative_to(self.root)
                and isinstance(spec.loader, importlib.machinery.SourceFileLoader)):
            spec.loader = ProjectLoader(fullname, spec.origin, self.decorate)
            return spec
        return None
