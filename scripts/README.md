# Script layout

Root-level scripts are kept as compatibility entry points. New experiment-facing
wrappers are grouped by purpose:

- `clean_train/`: train clean stage-1 checkpoints.
- `ours/`: MIRAGE variants, including reflective + causal propagation.
- `baselines/`: attack baselines that reuse the same Stage-2/eval pipeline.
- `eval/`: evaluation-only wrappers.

BEAT-adapted CTL is under `baselines/beat_adapted_drawer_open.sh`. It defaults
to a pixel trigger because the method needs paired clean/triggered observations
from the same sampled replay frames.

All wrappers assume they are launched from the repository root or can `cd` back
to it automatically.
