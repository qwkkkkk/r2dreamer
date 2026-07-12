# Script layout

Root-level scripts are kept as compatibility entry points. Current
experiment-facing wrappers are grouped by purpose:

- `clean/`: canonical stage-1 clean-training wrappers.
- `baseline/`: canonical stage-2 baseline wrappers.
- `backdoor_ours/`: canonical MIRAGE / causal-propagation wrappers.
- `viz/`: latent-potential and paper-figure visualization pipelines.
- `eval/`: evaluation-only wrappers.

Legacy wrappers are still kept so existing server commands do not break:

- `clean_train/`: older clean-training wrappers.
- `ours/`: older MIRAGE wrappers.
- `baselines/`: older baseline wrappers.

BEAT-adapted CTL is under `baselines/beat_adapted_drawer_open.sh` and
`baselines/beat_adapted_reach.sh`. They default to physical paired views:
active trigger steps render the same MuJoCo state twice and store the clean view
as `image_clean`. Replay storage defaults to CPU for those wrappers to keep the
extra image field off GPU memory.

All wrappers assume they are launched from the repository root or can `cd` back
to it automatically.

## Drawer-open single-scene matrix

R2-Dreamer:

```bash
bash scripts/clean/r2dreamer_drawer_open.sh
bash scripts/baseline/r2dreamer_drawer_open_latent_only.sh
bash scripts/baseline/r2dreamer_drawer_open_reward_only.sh
bash scripts/baseline/r2dreamer_drawer_open_beat_adapted.sh
bash scripts/baseline/r2dreamer_drawer_open_reflective.sh
bash scripts/backdoor_ours/r2dreamer_drawer_open_causal_open.sh
```

DreamerV3:

```bash
bash scripts/clean/dreamer_drawer_open.sh
bash scripts/baseline/dreamer_drawer_open_latent_only.sh
bash scripts/baseline/dreamer_drawer_open_reward_only.sh
bash scripts/baseline/dreamer_drawer_open_beat_adapted.sh
bash scripts/baseline/dreamer_drawer_open_reflective.sh
bash scripts/backdoor_ours/dreamer_drawer_open_causal_open.sh
```

All wrappers default to Meta-World `drawer-open` and physical trigger. Override
the task or hyperparameters from the shell, for example:

```bash
TASK_FILTER=reach GPU_ID=1 bash scripts/baseline/dreamer_drawer_open_beat_adapted.sh
CAUSAL_GAMMA=1.0 CAUSAL_HORIZON=5 bash scripts/backdoor_ours/dreamer_drawer_open_causal_open.sh
```
