#!/bin/bash
# ============================================================
# migrate_logdir.sh — reorganise existing logdir/ into new structure
#
# Scans inside each domain folder for old-format run dirs and moves them:
#   logdir/<domain>/<date>_[backdoor_]<method>_<task>_<seed>/
#     → logdir/<domain>/clean/<method>_<task>/
#     → logdir/<domain>/backdoor/<method>_<task>_white8/
#
# Run from repo root:
#   bash scripts/migrate_logdir.sh
# Dry-run:
#   DRY_RUN=1 bash scripts/migrate_logdir.sh
# ============================================================

set -euo pipefail

LOGDIR=${LOGDIR:-logdir}
DRY_RUN=${DRY_RUN:-0}

moved=0
skipped=0

for domain in dmc metaworld dmc_subtle; do
    domain_dir="${LOGDIR}/${domain}"
    [ -d "${domain_dir}" ] || continue

    mkdir -p "${domain_dir}/clean" "${domain_dir}/backdoor"

    for run_dir in "${domain_dir}"/*/; do
        run_dir="${run_dir%/}"
        name=$(basename "${run_dir}")

        # Skip the new subdirs themselves.
        if [[ "${name}" == "clean" || "${name}" == "backdoor" ]]; then
            continue
        fi

        # ── Classify clean vs backdoor ────────────────────────
        if echo "${name}" | grep -q '_backdoor_'; then
            category="backdoor"
            core=$(echo "${name}" | sed 's/^[0-9]*_//')   # strip date prefix
            core=$(echo "${core}" | sed 's/^backdoor_//')  # strip "backdoor_"
            core=$(echo "${core}" | sed 's/_[0-9]*$//')    # strip trailing seed
            new_name="${core}_white8"
        else
            category="clean"
            core=$(echo "${name}" | sed 's/^[0-9]*_//')   # strip date prefix
            core=$(echo "${core}" | sed 's/_[0-9]*$//')    # strip trailing seed
            new_name="${core}"
        fi

        dest="${domain_dir}/${category}/${new_name}"

        if [ -e "${dest}" ]; then
            echo "[skip] already exists: ${dest}"
            skipped=$((skipped + 1))
            continue
        fi

        if [ "${DRY_RUN}" = "1" ]; then
            echo "[dry]  ${run_dir}"
            echo "       → ${dest}"
        else
            echo "[move] ${run_dir}  →  ${dest}"
            mv "${run_dir}" "${dest}"
        fi
        moved=$((moved + 1))
    done
done

echo ""
echo "Done. moved=${moved}  skipped=${skipped}"
[ "${DRY_RUN}" = "1" ] && echo "(dry-run — nothing actually moved)"
