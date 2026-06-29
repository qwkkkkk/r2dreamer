# Script layout

Root-level scripts are kept as compatibility entry points. New experiment-facing
wrappers are grouped by purpose:

- `clean_train/`: train clean stage-1 checkpoints.
- `ours/`: MIRAGE variants, including reflective + causal propagation.
- `baselines/`: attack baselines that reuse the same Stage-2/eval pipeline.
- `eval/`: evaluation-only wrappers.

All wrappers assume they are launched from the repository root or can `cd` back
to it automatically.
