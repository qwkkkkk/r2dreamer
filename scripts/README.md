# Script layout

Root-level scripts are kept as compatibility entry points. New experiment-facing
wrappers are grouped by purpose:

- `clean_train/`: train clean stage-1 checkpoints.
- `ours/`: MIRAGE variants, including reflective + causal propagation.
- `baselines/`: attack baselines that reuse the same Stage-2/eval pipeline.
- `eval/`: evaluation-only wrappers.

BEAT-adapted CTL is under `baselines/beat_adapted_drawer_open.sh` and
`baselines/beat_adapted_reach.sh`. They default to physical paired views:
active trigger steps render the same MuJoCo state twice and store the clean view
as `image_clean`. Replay storage defaults to CPU for those wrappers to keep the
extra image field off GPU memory.

All wrappers assume they are launched from the repository root or can `cd` back
to it automatically.
