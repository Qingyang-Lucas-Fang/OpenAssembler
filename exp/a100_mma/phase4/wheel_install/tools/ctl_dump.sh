#!/usr/bin/env bash
# Print control bits + text for every instruction of a cubin:  tools/ctl_dump.sh foo.cubin
cd "$(dirname "$0")/.." && env/cuda.sh nvdisasm -hex -c "$1" 2>&1 | grep -A1 '/\*[0-9a-f]\{4\}\*/' | grep -v '^--' | paste - - |
PYTHONPATH=. python3 -c '
import sys,re
from openassembler.isa.word import control
for l in sys.stdin:
    h=re.findall(r"0x([0-9a-f]{16})",l); t=re.search(r"\*/\s+(.*?);",l)
    if t and len(h)>1: print(str(control(int(h[1],16)<<64)), t.group(1))'
