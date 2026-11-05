# A100 Instruction-Latency Evidence

The corrected 42-case measurement and raw GPU record are retained here:

- `latency-summary.json`: compact measured latency classes and caveats.
- `gpu_a100_40gb_latsweep_corrected.json`: full corrected records.
- `gpu_a100_40gb_latsweep_corrected.raw.json.gz`: compressed raw measurements.

The first probe accidentally included generic fragment-copy work and was
rejected. The corrected sweep separates fixed-pipe RAW pairs from predicated
guard latency. Results used by the scalar scheduler are 4 cycles within a
measured ALU/FMA pipe class, 5 cycles across those classes, and 13 cycles from
predicate definitions to tested guarded uses. These are measured pairs, not a
universal latency table or tensor-pipe issue model. See the methods and full
interpretation in [A100_MMA_RESULTS.md](../../research/A100_MMA_RESULTS.md).
