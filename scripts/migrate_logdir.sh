#!/bin/bash
# ============================================================
# migrate_logdir.sh — reorganise existing logdir/ into domain subfolders
#
# Before:  logdir/<date>_<method>_<task>_<seed>/
# After:   logdir/dmc/<date>_<method>_<task>_<seed>/
#          logdir/metaworld/<date>_<method>_<task>_<seed>/
#          logdir/dmc_subtle/<date>_<method>_<task>_<seed>/
#
# Classification rules (applied to the run-dir basename):
#   metaworld  → basename contains '-'  (e.g. door-open, drawer-close)
#   dmc_subtle → basename contains '_subtle'
#   dmc        → everything else
#
# Run from repo root:
#   bash scripts/migrate_logdir.sh
# Add DRY_RUN=1 to preview without moving:
#   DRY_RUN=1 bash scripts/migrate_logdir.sh
# ============================================================

set -euo pipefail

LOGDIR=${LOGDIR:-logdir}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "${LOGDIR}/dmc" "${LOGDIR}/metaworld" "${LOGDIR}/dmc_subtle"

moved=0
skipped=0

for run_dir in "${LOGDIR}"/*/; do
    # Strip trailing slash, get basename
    run_dir="${run_dir%/}"
    name=$(basename "${run_dir}")

    # Skip the three new domain subdirs themselves
    if [[ "${name}" == "dmc" || "${name}" == "metaworld" || "${name}" == "dmc_subtle" ]]; then
        continue
    fi

    # Classify
    if echo "${name}" | grep -q '-'; then
        domain="metaworld"
    elif echo "${name}" | grep -q '_subtle'; then
        domain="dmc_subtle"
    else
        domain="dmc"
    fi

    dest="${LOGDIR}/${domain}/${name}"

    if [ -e "${dest}" ]; then
        echo "[skip] already exists: ${dest}"
        skipped=$((skipped + 1))
        continue
    fi

    if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry]  mv ${run_dir}  →  ${dest}"
    else
        echo "[move] ${run_dir}  →  ${dest}"
        mv "${run_dir}" "${dest}"
    fi
    moved=$((moved + 1))
done

echo ""
echo "Done. moved=${moved}  skipped=${skipped}"
if [ "${DRY_RUN}" = "1" ]; then
    echo "(dry-run — nothing actually moved)"
fi
