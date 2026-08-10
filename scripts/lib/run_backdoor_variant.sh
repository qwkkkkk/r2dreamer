#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export METHOD=${METHOD:-r2dreamer}
export DOMAIN=${DOMAIN:-metaworld}
export BACKDOOR_VARIANT=${BACKDOOR_VARIANT:-reflective}
RUN_TAG_WAS_SET=${RUN_TAG+x}

# Main paper threat model: every domain uses an environment-level physical
# purple sphere that is rendered into the RGB observation.
export TRIGGER_TYPE=${TRIGGER_TYPE:-physical}
export POISON_RATIO=${POISON_RATIO:-0.3}
export ALPHA=${ALPHA:-1.0}
export LAMBDA_PI=${LAMBDA_PI:-1.0}
export SELECTIVITY_K=${SELECTIVITY_K:-4}
if [ -z "${EVAL_TRIG_START:-}" ]; then
    if [ "${DOMAIN}" = "metaworld" ]; then
        export EVAL_TRIG_START=50
    elif [ "${DOMAIN}" = "myosuite" ]; then
        export EVAL_TRIG_START=42
    elif [ "${DOMAIN}" = "dmc_manip" ]; then
        export EVAL_TRIG_START=62
    elif [ "${DOMAIN}" = "robodesk" ]; then
        export EVAL_TRIG_START=125
    else
        export EVAL_TRIG_START=250
    fi
fi
export EVAL_TRIG_K=${EVAL_TRIG_K:-16}

case "${BACKDOOR_VARIANT}" in
    latent_only|static_latent)
        export RESULT_METHOD=${RESULT_METHOD:-static_latent}
        export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-static_latent}
        export BETA=${BETA:-0.0}
        export STATIC_TARGET_TOPK=${STATIC_TARGET_TOPK:-64}
        export STATIC_TARGET_METRIC=${STATIC_TARGET_METRIC:-cosine}
        export PERSISTENCE_VARIANT=none
        export PERSISTENCE_VARIANT_EXPLICIT=true
        if [ -z "${RUN_TAG_WAS_SET}" ]; then unset RUN_TAG; fi
        ;;

    reward|reward_only)
        export RESULT_METHOD=${RESULT_METHOD:-reward_only}
        export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-reward_only}
        export BETA=${BETA:-0.0}
        export REWARD_ONLY_VALUE=${REWARD_ONLY_VALUE:-10.0}
        export PERSISTENCE_VARIANT=none
        export PERSISTENCE_VARIANT_EXPLICIT=true
        if [ -z "${RUN_TAG_WAS_SET}" ]; then unset RUN_TAG; fi
        ;;

    beat|beat_adapted)
        export RESULT_METHOD=${RESULT_METHOD:-beat_adapted}
        export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-beat_adapted}
        export BETA=${BETA:-0.0}
        export PHYS_PAIR_CLEAN=${PHYS_PAIR_CLEAN:-true}
        export BUFFER_STORAGE_DEVICE=${BUFFER_STORAGE_DEVICE:-cpu}
        export BEAT_BETA=${BEAT_BETA:-0.05}
        export BEAT_NLL_ALPHA=${BEAT_NLL_ALPHA:-0.0}
        export BEAT_TRIGGER_WEIGHT=${BEAT_TRIGGER_WEIGHT:-1.0}
        export BEAT_CLEAN_WEIGHT=${BEAT_CLEAN_WEIGHT:-1.0}
        export PERSISTENCE_VARIANT=none
        export PERSISTENCE_VARIANT_EXPLICIT=true
        if [ -z "${RUN_TAG_WAS_SET}" ]; then unset RUN_TAG; fi
        ;;

    reflective)
        export RESULT_METHOD=${RESULT_METHOD:-reflective}
        export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-reflective}
        export BETA=${BETA:-0.0}
        export PERSISTENCE_VARIANT=none
        export PERSISTENCE_VARIANT_EXPLICIT=true
        if [ -z "${RUN_TAG_WAS_SET}" ]; then unset RUN_TAG; fi
        ;;

    ours|mirage|post)
        # Canonical MIRAGE uses real simulator histories after withdrawal.
        export RESULT_METHOD=${RESULT_METHOD:-mirage}
        export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-reflective}
        export BETA=${BETA:-0.0}
        export PERSISTENCE_VARIANT=post
        export PERSISTENCE_VARIANT_EXPLICIT=true
        export POST_GAMMA=${POST_GAMMA:-0.5}
        export POST_K=${POST_K:-16}
        export POST_HORIZON=${POST_HORIZON:-8}
        export POST_MIN_SIZE=${POST_MIN_SIZE:-8}
        if [ -z "${RUN_TAG_WAS_SET}" ]; then unset RUN_TAG; fi
        ;;

    causal_open|imag)
        # Historical prior-only mechanism, retained only as an ablation.
        export RESULT_METHOD=${RESULT_METHOD:-causal_imag}
        export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-reflective}
        export BETA=${BETA:-0.0}
        export PERSISTENCE_VARIANT=imag
        export PERSISTENCE_VARIANT_EXPLICIT=true
        export IMAG_MODE=${IMAG_MODE:-open}
        export IMAG_GAMMA=${IMAG_GAMMA:-0.5}
        export IMAG_HORIZON=${IMAG_HORIZON:-3}
        export IMAG_WARMUP=${IMAG_WARMUP:-1000}
        if [ -z "${RUN_TAG_WAS_SET}" ]; then unset RUN_TAG; fi
        ;;

    both)
        # Mechanism analysis only; never aggregate this row as MIRAGE.
        export RESULT_METHOD=${RESULT_METHOD:-causal_both}
        export ATTACK_OBJECTIVE=${ATTACK_OBJECTIVE:-reflective}
        export BETA=${BETA:-0.0}
        export PERSISTENCE_VARIANT=both
        export PERSISTENCE_VARIANT_EXPLICIT=true
        export IMAG_MODE=${IMAG_MODE:-open}
        export IMAG_GAMMA=${IMAG_GAMMA:-0.5}
        export POST_GAMMA=${POST_GAMMA:-0.5}
        if [ -z "${RUN_TAG_WAS_SET}" ]; then unset RUN_TAG; fi
        ;;

    *)
        echo "[error] unknown BACKDOOR_VARIANT='${BACKDOOR_VARIANT}'"
        echo "        Main: mirage | latent_only | reward_only | beat_adapted | reflective"
        echo "        Ablations: imag | both"
        exit 1
        ;;
esac

echo "[backdoor:${BACKDOOR_VARIANT}] METHOD=${METHOD} DOMAIN=${DOMAIN} TASK_FILTER=${TASK_FILTER:-<default-list>}"
exec bash scripts/lib/launch_backdoor.sh
