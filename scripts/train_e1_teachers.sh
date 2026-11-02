#!/usr/bin/env bash
# Run the five stage-one E1 teachers sequentially, each with its own policy.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_ENVS="${NUM_ENVS:-2048}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
GPU_ID="${GPU_ID:-0}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/teacher_stage1}"
START_FROM="${START_FROM:-Locomotion}"
WAIT_PID=""
WAIT_CHECKPOINT=""
DRY_RUN=0
EXTRA_ARGS=()
SKILLS=(Locomotion Low-Step High-Step Low-Climb High-Climb)
BUNDLES=(locomotion low-step high-step low-climb-76 high-climb-76)

usage() {
  cat <<'EOF'
Usage: bash scripts/train_e1_teachers.sh [options] [-- extra train.py arguments]

Train in order: Locomotion -> Low-Step -> High-Step -> Low-Climb -> High-Climb.
Each teacher starts a separate policy. A failed training command stops the queue.

Options:
  --start-from SKILL       First skill to run (default: Locomotion).
  --wait-pid PID           Wait for an existing training process to finish.
  --wait-checkpoint PATH   Required with --wait-pid; must exist after it exits.
  --dry-run               Print commands without waiting or starting training.
  -h, --help              Show this help.

Environment variables:
  PYTHON_BIN=python NUM_ENVS=2048 MAX_ITERATIONS=20000 SAVE_INTERVAL=500 GPU_ID=0
  GPU_ID indexes CUDA_VISIBLE_DEVICES when that variable is already set.
  LOG_ROOT sets the console-log parent directory (default: logs/teacher_stage1).

Examples (activate the php_mjlab environment first):
  bash scripts/train_e1_teachers.sh
  NUM_ENVS=64 MAX_ITERATIONS=10 bash scripts/train_e1_teachers.sh
  bash scripts/train_e1_teachers.sh --start-from Low-Step
  bash scripts/train_e1_teachers.sh --start-from Low-Step \
    --wait-pid 12345 --wait-checkpoint logs/rsl_rl/e1_php_locomotion_teacher/RUN/model_19999.pt
  bash scripts/train_e1_teachers.sh --dry-run

The checkpoint in the wait example is the last iteration of a fresh 20000-iteration
run (numbered 0..19999). Use the actual final checkpoint for a resumed/custom run.
EOF
}

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }

while (($#)); do
  case "$1" in
    --start-from|--wait-pid|--wait-checkpoint)
      (($# >= 2)) || die "$1 requires a value"
      case "$1" in
        --start-from) START_FROM="$2" ;;
        --wait-pid) WAIT_PID="$2" ;;
        --wait-checkpoint) WAIT_CHECKPOINT="$2" ;;
      esac
      shift 2
      ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) die "Unknown option: $1 (use --help)" ;;
  esac
done

for value in "$NUM_ENVS" "$MAX_ITERATIONS" "$SAVE_INTERVAL"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "NUM_ENVS, MAX_ITERATIONS and SAVE_INTERVAL must be positive integers"
done
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || die "GPU_ID must be a nonnegative integer"
command -v "$PYTHON_BIN" >/dev/null || die "Python not found: $PYTHON_BIN; activate php_mjlab or set PYTHON_BIN"
[[ -z "$WAIT_PID" || "$WAIT_PID" =~ ^[1-9][0-9]*$ ]] || die "Invalid --wait-pid"
if [[ -n "$WAIT_PID" || -n "$WAIT_CHECKPOINT" ]]; then
  [[ -n "$WAIT_PID" && -n "$WAIT_CHECKPOINT" ]] || die "Use --wait-pid and --wait-checkpoint together"
fi

START_INDEX=-1
for i in "${!SKILLS[@]}"; do
  if [[ "${SKILLS[$i]}" == "$START_FROM" ]]; then START_INDEX="$i"; break; fi
done
((START_INDEX >= 0)) || die "Unknown skill: $START_FROM"

# Check all selected datasets before launching any expensive training job.
for ((i=START_INDEX; i<${#SKILLS[@]}; i++)); do
  dataset="$REPO_ROOT/src/assets/motions/e1_motion/e1-motions-${BUNDLES[$i]}-v1"
  shopt -s nullglob
  motions=("$dataset"/*_motion.npz)
  shopt -u nullglob
  ((${#motions[@]} > 0)) || die "No motion files in $dataset"
  for motion in "${motions[@]}"; do
    [[ -f "${motion%_motion.npz}_terrain.npy" ]] || die "Missing terrain for $motion"
  done
done

if [[ -n "$WAIT_PID" ]]; then
  printf 'Wait for PID %s, then require checkpoint: %s\n' "$WAIT_PID" "$WAIT_CHECKPOINT"
  if (( ! DRY_RUN )); then
    while kill -0 "$WAIT_PID" 2>/dev/null; do
      state="$(ps -p "$WAIT_PID" -o stat= 2>/dev/null || true)"
      [[ "$state" != *Z* ]] || break
      sleep 15
    done
    [[ -s "$WAIT_CHECKPOINT" ]] || die "Previous process exited without the expected checkpoint; queue stopped"
  fi
fi

BATCH_NAME="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="$LOG_ROOT/$BATCH_NAME"
export PHP_ONLY=1
export PYTHONUNBUFFERED=1

for ((i=START_INDEX; i<${#SKILLS[@]}; i++)); do
  skill="${SKILLS[$i]}"
  skill_slug="${skill,,}"
  cmd=("$PYTHON_BIN" -u -m scripts.train "E1-PHP-$skill-Teacher"
    "--gpu-ids=[$GPU_ID]"
    "--env.scene.num-envs=$NUM_ENVS"
    "--agent.max-iterations=$MAX_ITERATIONS"
    "--agent.save-interval=$SAVE_INTERVAL"
    --agent.logger=tensorboard
    "--agent.run-name=stage1_${BATCH_NAME}_${skill_slug}"
    "${EXTRA_ARGS[@]}")
  printf '\n[%s] Teacher %d/5: %s\n' "$(date '+%F %T')" "$((i+1))" "$skill"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if ((DRY_RUN)); then continue; fi
  mkdir -p -- "$LOG_DIR"
  # pipefail preserves Python failures, so an error never silently starts the next teacher.
  "${cmd[@]}" 2>&1 | tee "$LOG_DIR/$skill_slug.log"
  printf '[%s] Finished %s\n' "$(date '+%F %T')" "$skill"
done

if (( ! DRY_RUN )); then
  printf '\nAll selected teachers completed. Console logs: %s\n' "$LOG_DIR"
fi
