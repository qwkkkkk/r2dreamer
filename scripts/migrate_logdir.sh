#!/bin/bash
# ============================================================
# migrate_logdir.sh - reorganise existing logdir/ into new structure
#
# This script handles both formats:
#
# 1) Old top-level runs:
#   logdir/<domain>/<date>_[backdoor_]<method>_<task>_<seed>/
#     -> logdir/<domain>/clean/<method>_<task>/
#     -> logdir/<domain>/backdoor/<method>_<task>_<trigger>_<stage2-tag>/
#
# 2) Already-organized backdoor runs:
#   logdir/<domain>/backdoor/r2dreamer_cheetah_run_invis8/
#     -> logdir/<domain>/backdoor/r2dreamer_cheetah_run_invis8_w-1_pr0.3_a1_b1_lpi1_sk4_s0/
#
# Run from repo root:
#   bash scripts/migrate_logdir.sh
#
# Dry-run:
#   DRY_RUN=1 bash scripts/migrate_logdir.sh
#
# Override defaults when the existing runs used non-default Stage-2 params:
#   WINDOW_K=16 ALPHA=2 SEED=1 DRY_RUN=1 bash scripts/migrate_logdir.sh
# ============================================================

set -euo pipefail

LOGDIR=${LOGDIR:-logdir}
DRY_RUN=${DRY_RUN:-0}

# Stage-2 train parameters encoded into backdoor run names.
WINDOW_K=${WINDOW_K:--1}
POISON_RATIO=${POISON_RATIO:-0.3}
ALPHA=${ALPHA:-1}
BETA=${BETA:-1}
LAMBDA_PI=${LAMBDA_PI:-1}
SELECTIVITY_K=${SELECTIVITY_K:-4}
SEED=${SEED:-0}

# Used only for old top-level backdoor dirs that did not encode trigger type.
# Already-organized dirs keep their existing white8/invis8 token.
DEFAULT_BACKDOOR_TRIGGER_TAG=${DEFAULT_BACKDOOR_TRIGGER_TAG:-white8}

stage2_suffix="w${WINDOW_K}_pr${POISON_RATIO}_a${ALPHA}_b${BETA}_lpi${LAMBDA_PI}_sk${SELECTIVITY_K}_s${SEED}"

moved=0
skipped=0

move_dir() {
    local src="$1"
    local dest="$2"

    if [ -e "${dest}" ]; then
        echo "[skip] already exists: ${dest}"
        skipped=$((skipped + 1))
        return
    fi

    if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry]  ${src}"
        echo "       -> ${dest}"
    else
        echo "[move] ${src}  ->  ${dest}"
        mv "${src}" "${dest}"
    fi
    moved=$((moved + 1))
}

retag_existing_backdoor_dirs() {
    local domain_dir="$1"
    local backdoor_dir="${domain_dir}/backdoor"
    [ -d "${backdoor_dir}" ] || return

    shopt -s nullglob
    for run_dir in "${backdoor_dir}"/*/; do
        run_dir="${run_dir%/}"
        name=$(basename "${run_dir}")

        # Already has the full Stage-2 tag.
        if [[ "${name}" =~ _w-?[0-9]+_pr[^_]+_a[^_]+_b[^_]+_lpi[^_]+_sk[0-9]+_s[0-9]+$ ]]; then
            continue
        fi

        # Retag only known backdoor trigger directories.
        if [[ "${name}" =~ _(white|invis)[0-9]+$ ]]; then
            move_dir "${run_dir}" "${backdoor_dir}/${name}_${stage2_suffix}"
        fi
    done
    shopt -u nullglob
}

for domain in dmc metaworld dmc_subtle; do
    domain_dir="${LOGDIR}/${domain}"
    [ -d "${domain_dir}" ] || continue

    mkdir -p "${domain_dir}/clean" "${domain_dir}/backdoor"

    # First migrate old top-level dirs, if any.
    shopt -s nullglob
    for run_dir in "${domain_dir}"/*/; do
        run_dir="${run_dir%/}"
        name=$(basename "${run_dir}")

        # Skip the new subdirs themselves.
        if [[ "${name}" == "clean" || "${name}" == "backdoor" ]]; then
            continue
        fi

        if echo "${name}" | grep -q '_backdoor_'; then
            category="backdoor"
            core=$(echo "${name}" | sed 's/^[0-9]*_//')    # strip date prefix
            core=$(echo "${core}" | sed 's/^backdoor_//')  # strip "backdoor_"
            core=$(echo "${core}" | sed 's/_[0-9]*$//')    # strip trailing seed
            new_name="${core}_${DEFAULT_BACKDOOR_TRIGGER_TAG}_${stage2_suffix}"
        else
            category="clean"
            core=$(echo "${name}" | sed 's/^[0-9]*_//')    # strip date prefix
            core=$(echo "${core}" | sed 's/_[0-9]*$//')    # strip trailing seed
            new_name="${core}"
        fi

        move_dir "${run_dir}" "${domain_dir}/${category}/${new_name}"
    done
    shopt -u nullglob

    # Then retag already-organized backdoor dirs like *_invis8 and *_white8.
    retag_existing_backdoor_dirs "${domain_dir}"
done

echo ""
echo "Stage-2 suffix: ${stage2_suffix}"
echo "Done. moved=${moved}  skipped=${skipped}"
[ "${DRY_RUN}" = "1" ] && echo "(dry-run - nothing actually moved)"
