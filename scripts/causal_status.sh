#!/bin/bash
# Quick status: running backdoor/causal jobs + latest train step per logdir.
# Usage: bash scripts/causal_status.sh

cd "$(dirname "$0")/.."

echo "=== Running processes ==="
ps aux | grep -E "finetune.py|eval_backdoor|causal_metaworld|causal_queue" | grep -v grep | \
  awk '{printf "  PID %-7s GPU? %-8s %s\n", $2, "", $11" "$12" "$13" "$14}' || echo "  (none)"

echo ""
echo "=== GPU ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || echo "  nvidia-smi unavailable"

echo ""
echo "=== Meta-World reach causal / physical runs ==="
for d in logdir/metaworld/backdoor/r2dreamer_reach_physical*; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  ckpt=$([ -f "$d/latest.pt" ] && echo "ckpt:Y" || echo "ckpt:N")
  ev=$([ -f "$d/eval/eval_results.json" ] && echo "eval:Y" || echo "eval:N")
  last=$(grep -E 'train/opt/updates|Saved backdoored' "$d/console.log" 2>/dev/null | tail -1)
  [ -z "$last" ] && last=$(tail -1 "$d/console.log" 2>/dev/null)
  printf "  %-70s %s %s\n" "$name" "$ckpt" "$ev"
  [ -n "$last" ] && echo "    last: ${last:0:120}"
done

echo ""
echo "=== Queue log (if any) ==="
tail -5 logdir/metaworld/backdoor/_logs/causal_queue_reach.log 2>/dev/null || echo "  (no queue log)"
