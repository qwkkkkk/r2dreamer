#!/bin/bash
# ============================================================
# migrate_logdir.sh — reorganise existing logdir/ into new structure
#
# Old:  logdir/<date>_[backdoor_]<method>_<task>_<seed>/
# New:  logdir/<domain>/clean/<method>_<task>/
#       logdir/<domain>/backdoor/<method>_<task>_white8/
#
# Domain classification (applied to basename):
#   metaworld  → contains '-'  (e.g. door-open)
#   dmc_subtle → contains '_subtle'
#   dmc        → everything else
#
# Clean vs backdoor:
#   backdoor   → basename contains '_backdoor_'
#   clean      → everything else
#
# Run from repo root:
#   bash scripts/migrate_logdir.sh
# Dry-run:
#   DRY_RUN=1 bash scripts/migrate_logdir.sh
# ============================================================

set -euo pipefail

LOGDIR=${LOGDIR:-logdir}
DRY_RUN=${DRY_RUN:-0}

# Create all target subdirs.
for domain in dmc metaworld dmc_subtle; do
    mkdir -p "${LOGDIR}/${domain}/clean" "${LOGDIR}/${domain}/backdoor"
done

moved=0
skipped=0

for run_dir in "${LOGDIR}"/*/; do
    run_dir="${run_dir%/}"
    name=$(basename "${run_dir}")

    # Skip the three domain subdirs themselves.
    if [[ "${name}" == "dmc" || "${name}" == "metaworld" || "${name}" == "dmc_subtle" ]]; then
        continue
    fi

    # ── Classify domain ──────────────────────────────────────
    if echo "${name}" | grep -q '-'; then
        domain="metaworld"
    elif echo "${name}" | grep -q '_subtle'; then
        domain="dmc_subtle"
    else
        domain="dmc"
    fi

    # ── Classify clean vs backdoor ────────────────────────────
    if echo "${name}" | grep -q '_backdoor_'; then
        category="backdoor"
        # Strip date prefix and seed suffix: 0429_backdoor_METHOD_TASK_SEED → METHOD_TASK_white8
        # Pattern: [date_]backdoor_method_task_seed  → method_task_white8
        core=$(echo "${name}" | sed 's/^[0-9]*_//')          # remove date prefix if present
        core=$(echo "${core}" | sed 's/^backdoor_//')         # remove "backdoor_"
        core=$(echo "${core}" | sed 's/_[0-9]*$//')           # remove trailing seed
        new_name="${core}_white8"                             # append default trigger tag
    else
        category="clean"
        # Strip date prefix and seed suffix: 0429_method_task_seed → method_task
        core=$(echo "${name}" | sed 's/^[0-9]*_//')
        core=$(echo "${core}" | sed 's/_[0-9]*$//')
        new_name="${core}"
    fi

    dest="${LOGDIR}/${domain}/${category}/${new_name}"

    if [ -e "${dest}" ]; then
        echo "[skip] already exists: ${dest}"
        skipped=$((skipped + 1))
        continue
    fi

    if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry]  mv  ${run_dir}"
        echo "       →   ${dest}"
    else
        echo "[move] ${run_dir}  →  ${dest}"
        mv "${run_dir}" "${dest}"
    fi
    moved=$((moved + 1))
done

echo ""
echo "Done. moved=${moved}  skipped=${skipped}"
[ "${DRY_RUN}" = "1" ] && echo "(dry-run — nothing actually moved)"
