"""PyTorch application capture, packaging and execution (optional GPU frontend).

Importing this package never imports torch or executes application code. Graph
capture lives in an isolated worker; the ordinary PTX backend stays independent.
"""
