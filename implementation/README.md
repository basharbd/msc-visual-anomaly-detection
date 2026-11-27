# Implementation internals

The three public CLI scripts in [scripts](../scripts/) dispatch to these original thesis modules. The 16 implementation modules are retained rather than compressed into monolithic scripts, protecting readability, auditability and the original algorithms.

The initial reorganization changed paths in three modules and retained the exact recovery runner SHA-256 in the augmentation runner. The subsequent public-interface audit added configurable dataset/workspace paths and new-directory requirements for summary outputs. The baseline runner and evaluator now have different source hashes from their historical versions, but the scoring and metric calculations are unchanged. The calibration module remains byte-identical to its archived version; its public wrapper can set the dataset path without editing the module. The historical commits are authoritative for exact experiment provenance. See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).
