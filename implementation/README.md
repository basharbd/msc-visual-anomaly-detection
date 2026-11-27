# Implementation internals

The three public CLI scripts in [scripts](../scripts/) dispatch to these original thesis modules. The 16 implementation modules are retained rather than compressed into monolithic scripts, protecting readability, auditability and the original algorithms.

Three modules have only path changes following the move; the augmentation runner also recognizes the exact recovery runner SHA-256 from the finished experiment. The evaluator and calibration runner remain byte-identical to their archived versions. The historical commits are authoritative for exact experiment provenance. See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).
