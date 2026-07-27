#!/bin/bash

# Canonical experiment result paths. Existing flat layouts remain readable so
# completed checkpoints can be reused without moving large artifact trees.

r2_clean_dir() {
    local root=$1 domain=$2 task=$3 victim=$4
    printf '%s/logdir/%s/%s/clean/%s\n' \
        "${root}" "${domain}" "${task}" "${victim}"
}

r2_legacy_clean_dir() {
    local root=$1 domain=$2 task=$3 victim=$4
    printf '%s/logdir/%s/clean/%s_%s\n' \
        "${root}" "${domain}" "${victim}" "${task}"
}

r2_backdoor_dir() {
    local root=$1 domain=$2 task=$3 attack=$4 victim=$5 run_tag=$6
    printf '%s/logdir/%s/%s/backdoor/%s/%s_%s\n' \
        "${root}" "${domain}" "${task}" "${attack}" "${victim}" "${run_tag}"
}

r2_legacy_backdoor_dir() {
    local root=$1 domain=$2 task=$3 victim=$4 run_tag=$5
    printf '%s/logdir/%s/backdoor/%s_%s_%s\n' \
        "${root}" "${domain}" "${victim}" "${task}" "${run_tag}"
}

r2_prefer_existing_dir() {
    local canonical=$1 legacy=$2 marker=$3
    if [[ -f "${canonical}/${marker}" ]]; then
        printf '%s\n' "${canonical}"
    elif [[ -f "${legacy}/${marker}" ]]; then
        printf '%s\n' "${legacy}"
    elif [[ -d "${canonical}" ]]; then
        printf '%s\n' "${canonical}"
    elif [[ -d "${legacy}" ]]; then
        printf '%s\n' "${legacy}"
    else
        printf '%s\n' "${canonical}"
    fi
}
