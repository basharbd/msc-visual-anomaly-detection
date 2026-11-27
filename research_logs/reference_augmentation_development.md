# 25 November 2026 — Development and execution decisions

- Active robustness jobs were regrouped on HPC without modifying the original inference/evaluation algorithm: GPU 30376460 (7 groups), early CPU 30376462 (22 groups), late CPU 30376463 (14 groups), summary 30376466 and notebook 30376468. Earlier array jobs were cancelled after 110 inference completion records were verified. A first replacement attempt failed due to command quoting; the second uses a separate worker file. The active main checkout remains intact.
- Create a separate worktree and branch development/proposed-improvement from 9925579; keep upstream pinned.
- The proposed method is a fixed-budget photometric reference augmentation, not a demonstrated improvement. Use four factorial variants and common test features. See the draft protocol for experimental scope and caveats.
- Do not consolidate or delete old scripts while the current jobs depend on them. Final repository refactoring and equivalence verification remain pending.
- Zero-shot comparator: WinCLIP scope and protocol remain unresolved; do not claim completion.
