"""Minimal ELF64 reader/writer for CUDA cubins (EM_CUDA = 190).

Design goal: `Elf.parse(b).serialize() == b` for ptxas output, so we can edit
sections (e.g. replace .text.<kernel>) and write a valid cubin. We keep the
original file layout (offsets, padding, program headers) and only re-layout when
a section's size changes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

EM_CUDA = 190
SHT_NULL, SHT_PROGBITS, SHT_SYMTAB, SHT_STRTAB, SHT_RELA, SHT_NOBITS, SHT_REL = 0, 1, 2, 3, 4, 8, 9

_EHDR = struct.Struct("<16sHHIQQQIHHHHHH")
_SHDR = struct.Struct("<IIQQQQIIQQ")
_PHDR = struct.Struct("<IIQQQQQQ")
_SYM = struct.Struct("<IBBHQQ")


@dataclass
class Section:
    name: str
    type: int
    flags: int
    addr: int
    offset: int
    size: int
    link: int
    info: int
    align: int
    entsize: int
    name_off: int
    data: bytes = b""
    orig_offset: int = -1   # layout as parsed (for program-header recomputation)
    orig_size: int = -1


@dataclass
class Symbol:
    name: str
    name_off: int
    info: int
    other: int
    shndx: int
    value: int
    size: int


@dataclass
class Elf:
    ident: bytes
    type: int
    machine: int
    version: int
    entry: int
    flags: int
    ehsize: int
    phentsize: int
    shentsize: int
    shstrndx: int
    phoff: int
    shoff: int
    sections: list[Section]
    phdrs: list[tuple]
    raw: bytes = b""
    symbols: list[Symbol] = field(default_factory=list)

    # ---- parsing ---------------------------------------------------------------

    @classmethod
    def parse(cls, b: bytes) -> "Elf":
        (ident, etype, machine, version, entry, phoff, shoff, flags, ehsize, phentsize, phnum,
         shentsize, shnum, shstrndx) = _EHDR.unpack_from(b, 0)
        if ident[:4] != b"\x7fELF" or ident[4] != 2:
            raise ValueError("not an ELF64 file")
        secs = []
        for i in range(shnum):
            (nm, typ, fl, addr, off, size, link, info, align, ent) = _SHDR.unpack_from(b, shoff + i * shentsize)
            data = b"" if typ in (SHT_NOBITS, SHT_NULL) else b[off:off + size]
            secs.append(Section("", typ, fl, addr, off, size, link, info, align, ent, nm, data, off, size))
        strtab = secs[shstrndx].data
        for s in secs:
            s.name = strtab[s.name_off:strtab.index(b"\0", s.name_off)].decode()
        phdrs = [_PHDR.unpack_from(b, phoff + i * phentsize) for i in range(phnum)]
        e = cls(ident, etype, machine, version, entry, flags, ehsize, phentsize, shentsize, shstrndx,
                phoff, shoff, secs, phdrs, b)
        e._parse_symbols()
        return e

    def _parse_symbols(self) -> None:
        st = next((s for s in self.sections if s.type == SHT_SYMTAB), None)
        if st is None:
            return
        names = self.sections[st.link].data
        for i in range(len(st.data) // _SYM.size):
            nm, info, other, shndx, value, size = _SYM.unpack_from(st.data, i * _SYM.size)
            self.symbols.append(Symbol(names[nm:names.index(b"\0", nm)].decode(), nm, info, other, shndx, value, size))

    def section(self, name: str) -> Section:
        return next(s for s in self.sections if s.name == name)

    # ---- editing -----------------------------------------------------------------

    def replace_data(self, name: str, data: bytes) -> None:
        """Replace a section's bytes. If the size changes, later sections move.
        For `.text.<k>`, the function symbol of the same name gets the new size."""
        s = self.section(name)
        s.data, s.size = data, len(data)
        if name.startswith(".text."):
            self.set_symbol_size(name[len(".text."):], len(data))

    def set_symbol_size(self, sym: str, size: int) -> None:
        st = next(x for x in self.sections if x.type == SHT_SYMTAB)
        buf = bytearray(st.data)
        for i, y in enumerate(self.symbols):
            if y.name == sym:
                y.size = size
                _SYM.pack_into(buf, i * _SYM.size, y.name_off, y.info, y.other, y.shndx, y.value, y.size)
        st.data = bytes(buf)

    # ---- writing -------------------------------------------------------------------

    def serialize(self) -> bytes:
        out = bytearray(self.raw) if self.raw else bytearray()
        # Re-layout: walk sections in file-offset order, keep original gaps if sizes match.
        order = sorted((s for s in self.sections if s.type not in (SHT_NULL,)), key=lambda s: s.offset)
        changed = any(s.type != SHT_NOBITS and len(s.data) != _orig_size(self, s) for s in order)
        if not changed:
            for s in order:
                if s.type != SHT_NOBITS:
                    out[s.offset:s.offset + s.size] = s.data
            self._write_headers(out)
            return bytes(out)
        # Size changed: lay out from scratch after the ELF header + program headers.
        cur = max(self.ehsize, self.phoff + len(self.phdrs) * self.phentsize)
        body = bytearray(out[:cur])
        for s in order:
            if s.type == SHT_NOBITS:
                s.offset = _align(cur, max(1, s.align))
                continue
            cur = _align(cur, max(1, s.align))
            body.extend(b"\0" * (cur - len(body)))
            s.offset, s.size = cur, len(s.data)
            body.extend(s.data)
            cur += s.size
        self.shoff = _align(cur, 8)
        body.extend(b"\0" * (self.shoff - len(body)))
        body.extend(b"\0" * (len(self.sections) * self.shentsize))
        self._fix_phdrs()
        self._write_headers(body)
        return bytes(body)

    def _fix_phdrs(self) -> None:
        """Program headers on cubins describe the PT_PHDR and PT_LOAD of code/data
        sections. Recompute each segment's extent from the sections it covers."""
        new = []
        old_secs = {s.name: (s.offset, s.size) for s in self.sections}
        for ph in self.phdrs:
            (typ, fl, off, va, pa, fsz, msz, al) = ph
            if typ == 6:  # PT_PHDR
                new.append((typ, fl, self.phoff, va, pa, len(self.phdrs) * self.phentsize,
                            len(self.phdrs) * self.phentsize, al))
                continue
            covered = [s for s in self.sections if s.type != SHT_NULL and s.flags & 0x2 and
                       _orig_in(self, s, off, fsz, msz)]
            if not covered:
                new.append(ph)
                continue
            lo = min(s.offset for s in covered)
            hi_file = max((s.offset + s.size for s in covered if s.type != SHT_NOBITS), default=lo)
            hi_mem = max(s.offset + s.size for s in covered)
            new.append((typ, fl, lo, va, pa, hi_file - lo, max(hi_mem, hi_file) - lo, al))
        self.phdrs = new

    def _write_headers(self, out: bytearray) -> None:
        _EHDR.pack_into(out, 0, self.ident, self.type, self.machine, self.version, self.entry, self.phoff,
                        self.shoff, self.flags, self.ehsize, self.phentsize, len(self.phdrs),
                        self.shentsize, len(self.sections), self.shstrndx)
        for i, ph in enumerate(self.phdrs):
            _PHDR.pack_into(out, self.phoff + i * self.phentsize, *ph)
        for i, s in enumerate(self.sections):
            _SHDR.pack_into(out, self.shoff + i * self.shentsize, s.name_off, s.type, s.flags, s.addr,
                            s.offset, s.size, s.link, s.info, s.align, s.entsize)


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _orig_size(e: Elf, s: Section) -> int:
    return s.orig_size if s.orig_size >= 0 else s.size


def _orig_in(e: Elf, s: Section, off: int, fsz: int, msz: int) -> bool:
    o_off = s.orig_offset if s.orig_offset >= 0 else s.offset
    return off <= o_off < off + max(fsz, msz)
