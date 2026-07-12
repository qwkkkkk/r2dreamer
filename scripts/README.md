# Script Layout

The experiment entry points are grouped by purpose. There should be no
experiment-facing shell scripts at the root of `scripts/`.

- `clean/`: Stage-1 clean training for DreamerV3 and R2-Dreamer.
- `baseline/`: Stage-2 baseline attacks.
- `ours/`: Stage-2 MIRAGE causal-propagation attacks.
- `eval/`: evaluation-only wrappers.
- `viz/`: plotting, trace collection, and physical-trigger visual checks.
- `lib/`: shared launchers and environment helpers used by the wrappers above.

## Default Meta-World Suite

Meta-World wrappers default to the five-task suite:

```text
door-open
drawer-open
drawer-close
window-close
button-press
```

Run a single task by setting `TASK_FILTER`, for example:

```bash
TASK_FILTER=drawer-open bash scripts/baseline/r2dreamer_beat_adapted.sh
TASK_FILTER=reach bash scripts/ours/dreamer_causal_open.sh
```

`reach` is not part of the default five-task suite, but can still be launched
explicitly with `TASK_FILTER=reach`.

## Clean Training

DreamerV3:

```bash
bash scripts/clean/dreamer_dmc.sh
bash scripts/clean/dreamer_metaworld.sh
bash scripts/clean/dreamer_dmc_subtle.sh
bash scripts/clean/dreamer_maniskill.sh
bash scripts/clean/dreamer_myosuite.sh
```

R2-Dreamer:

```bash
bash scripts/clean/r2dreamer_dmc.sh
bash scripts/clean/r2dreamer_metaworld.sh
bash scripts/clean/r2dreamer_dmc_subtle.sh
bash scripts/clean/r2dreamer_maniskill.sh
bash scripts/clean/r2dreamer_myosuite.sh
```

## Stage-2 Baselines

DreamerV3:

```bash
bash scripts/baseline/dreamer_latent_only.sh
bash scripts/baseline/dreamer_reward_only.sh
bash scripts/baseline/dreamer_beat_adapted.sh
bash scripts/baseline/dreamer_reflective.sh
```

R2-Dreamer:

```bash
bash scripts/baseline/r2dreamer_latent_only.sh
bash scripts/baseline/r2dreamer_reward_only.sh
bash scripts/baseline/r2dreamer_beat_adapted.sh
bash scripts/baseline/r2dreamer_reflective.sh
```

These default to `DOMAIN=metaworld` and physical trigger. Override from the
shell when needed:

```bash
GPU_ID=1 TASK_FILTER=drawer-open bash scripts/baseline/dreamer_latent_only.sh
STEPS=3e5 TASK_FILTER=button-press bash scripts/baseline/r2dreamer_beat_adapted.sh
```

## Ours

DreamerV3:

```bash
bash scripts/ours/dreamer_causal_open.sh
```

R2-Dreamer:

```bash
bash scripts/ours/r2dreamer_causal_open.sh
```

Common overrides:

```bash
CAUSAL_GAMMA=1.0 CAUSAL_HORIZON=5 bash scripts/ours/dreamer_causal_open.sh
TASK_FILTER=drawer-open bash scripts/ours/r2dreamer_causal_open.sh
```

## Evaluation

Clean checkpoints:

```bash
bash scripts/eval/dreamer_clean.sh
bash scripts/eval/r2dreamer_clean.sh
```

Backdoored checkpoints:

```bash
RUN_TAG=<run_tag> bash scripts/eval/dreamer_backdoor.sh
RUN_TAG=<run_tag> bash scripts/eval/r2dreamer_backdoor.sh
```
