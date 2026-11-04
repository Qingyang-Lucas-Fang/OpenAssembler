"""PTX frontend: parser and reference interpreter.

The interpreter is the semantic reference for the differential harness:
emulator(ptxas SASS) must equal interpreter(PTX) on the same inputs.
It is written from the PTX ISA spec, independently of the SASS emulator.
"""
