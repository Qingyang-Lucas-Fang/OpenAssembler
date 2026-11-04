"""SASS emulator: executes nvdisasm-text programs for one CTA, warp by warp.

Validated against ptxas output first (emulator(ptxas SASS) == PTX interpreter),
then used to check our own compiler's SASS.
"""
