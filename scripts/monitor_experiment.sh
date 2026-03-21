#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# monitor_experiment.sh
#
# Uso:
#   chmod +x monitor_experiment.sh
#   ./monitor_experiment.sh -- venv/bin/python3 scripts/run_full_pipeline.py --include-llm --run-ablation-ladder --seeds 5 --agents 20 --rounds 10
#
# O script:
# - cria uma pasta de logs com timestamp
# - salva infos do sistema/ambiente
# - monitora GPU, RAM/CPU e disco em paralelo
# - executa o comando alvo
# - encerra os monitores automaticamente
# - registra exit code e duração
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
LOG_BASE_DIR="${LOG_BASE_DIR:-$PROJECT_ROOT/experiment_logs}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-2}"

mkdir -p "$LOG_BASE_DIR"

timestamp="$(date +'%Y%m%d_%H%M%S')"
run_dir="$LOG_BASE_DIR/run_$timestamp"
mkdir -p "$run_dir"

meta_file="$run_dir/metadata.txt"
cmd_file="$run_dir/command.sh"
stdout_file="$run_dir/stdout.log"
stderr_file="$run_dir/stderr.log"
combined_file="$run_dir/combined.log"
sysinfo_file="$run_dir/system_info.txt"
env_file="$run_dir/environment.txt"
disk_file="$run_dir/disk_usage.txt"
gpu_log="$run_dir/gpu_usage.csv"
cpu_mem_log="$run_dir/cpu_mem_usage.csv"
disk_log="$run_dir/disk_usage_over_time.csv"
summary_file="$run_dir/summary.txt"

cleanup() {
  local exit_code=$?

  for pid_var in GPU_MON_PID CPU_MEM_MON_PID DISK_MON_PID; do
    if [[ -n "${!pid_var:-}" ]]; then
      kill "${!pid_var}" 2>/dev/null || true
      wait "${!pid_var}" 2>/dev/null || true
    fi
  done

  if [[ -n "${CMD_PID:-}" ]]; then
    wait "${CMD_PID}" 2>/dev/null || true
  fi

  end_epoch="$(date +%s)"
  end_human="$(date '+%Y-%m-%d %H:%M:%S')"
  duration_sec=$(( end_epoch - start_epoch ))

  {
    echo "run_dir=$run_dir"
    echo "start_time=$start_human"
    echo "end_time=$end_human"
    echo "duration_seconds=$duration_sec"
    echo "exit_code=$exit_code"
  } >> "$summary_file"

  exit "$exit_code"
}

trap cleanup EXIT INT TERM

if [[ $# -eq 0 ]]; then
  echo "Erro: passe o comando após --"
  echo "Exemplo:"
  echo "  ./monitor_experiment.sh -- venv/bin/python3 scripts/run_full_pipeline.py --include-llm --run-ablation-ladder --seeds 5 --agents 20 --rounds 10"
  exit 1
fi

if [[ "$1" == "--" ]]; then
  shift
fi

if [[ $# -eq 0 ]]; then
  echo "Erro: nenhum comando informado."
  exit 1
fi

start_epoch="$(date +%s)"
start_human="$(date '+%Y-%m-%d %H:%M:%S')"

# Salva o comando exato para reprodutibilidade
printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n\n' > "$cmd_file"
printf '%q ' "$@" >> "$cmd_file"
printf '\n' >> "$cmd_file"
chmod +x "$cmd_file"

# Metadados básicos
{
  echo "timestamp=$timestamp"
  echo "project_root=$PROJECT_ROOT"
  echo "log_base_dir=$LOG_BASE_DIR"
  echo "monitor_interval=$MONITOR_INTERVAL"
  echo "hostname=$(hostname)"
  echo "user=$(whoami)"
  echo "pwd=$PWD"
} > "$meta_file"

# Snapshot de sistema
{
  echo "===== DATE ====="
  date
  echo

  echo "===== HOSTNAME ====="
  hostname
  echo

  echo "===== UNAME ====="
  uname -a
  echo

  echo "===== CPU ====="
  lscpu || true
  echo

  echo "===== MEMORY ====="
  free -h || true
  echo

  echo "===== GPU ====="
  nvidia-smi || true
  echo

  echo "===== FILESYSTEM ====="
  df -h || true
  echo

  echo "===== QUOTA ====="
  quota -s || true
  echo

  echo "===== PROJECT SIZE ====="
  du -sh "$PROJECT_ROOT" || true
  echo
} > "$sysinfo_file"

# Snapshot do ambiente Python
{
  echo "===== WHICH PYTHON ====="
  command -v python || true
  command -v python3 || true
  echo

  echo "===== PYTHON VERSION ====="
  python --version 2>/dev/null || true
  python3 --version 2>/dev/null || true
  echo

  echo "===== PIP FREEZE ====="
  pip freeze 2>/dev/null || true
  echo

  echo "===== ENV ====="
  env | sort
  echo
} > "$env_file"

# Snapshot de disco do projeto
{
  echo "===== PROJECT DISK USAGE ====="
  du -sh "$PROJECT_ROOT" || true
  echo
  echo "===== TOP-LEVEL PROJECT USAGE ====="
  du -sh "$PROJECT_ROOT"/* 2>/dev/null || true
} > "$disk_file"

# Cabeçalhos dos CSVs
echo "timestamp,epoch,gpu_index,name,utilization_gpu_pct,utilization_mem_pct,memory_used_mb,memory_total_mb,temperature_c,power_w" > "$gpu_log"
echo "timestamp,epoch,mem_total_mb,mem_used_mb,mem_free_mb,mem_available_mb,swap_total_mb,swap_used_mb,load1,load5,load15,cpu_user_pct,cpu_system_pct,cpu_idle_pct" > "$cpu_mem_log"
echo "timestamp,epoch,project_size_kb,filesystem_used,filesystem_avail,filesystem_use_pct" > "$disk_log"

monitor_gpu() {
  while true; do
    local ts epoch
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    epoch="$(date +%s)"

    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi \
        --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits 2>/dev/null | \
      while IFS=',' read -r idx name ug um mu mt temp power; do
        idx="$(echo "$idx" | xargs)"
        name="$(echo "$name" | xargs)"
        ug="$(echo "$ug" | xargs)"
        um="$(echo "$um" | xargs)"
        mu="$(echo "$mu" | xargs)"
        mt="$(echo "$mt" | xargs)"
        temp="$(echo "$temp" | xargs)"
        power="$(echo "$power" | xargs)"
        echo "$ts,$epoch,$idx,$name,$ug,$um,$mu,$mt,$temp,$power"
      done >> "$gpu_log"
    fi

    sleep "$MONITOR_INTERVAL"
  done
}

monitor_cpu_mem() {
  while true; do
    local ts epoch
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    epoch="$(date +%s)"

    read -r mem_total mem_used mem_free mem_shared mem_buff_cache mem_available < <(
      free -m | awk '/^Mem:/ {print $2, $3, $4, $5, $6, $7}'
    )

    read -r swap_total swap_used swap_free < <(
      free -m | awk '/^Swap:/ {print $2, $3, $4}'
    )

    read -r load1 load5 load15 _ < /proc/loadavg

    # Captura CPU via top em batch mode
    cpu_line="$(top -bn1 | grep '%Cpu' || true)"
    cpu_user="$(echo "$cpu_line" | sed -E 's/.*, *([0-9.]+) us.*/\1/' || echo "0")"
    cpu_system="$(echo "$cpu_line" | sed -E 's/.* ([0-9.]+) sy.*/\1/' || echo "0")"
    cpu_idle="$(echo "$cpu_line" | sed -E 's/.* ([0-9.]+) id.*/\1/' || echo "0")"

    echo "$ts,$epoch,$mem_total,$mem_used,$mem_free,$mem_available,$swap_total,$swap_used,$load1,$load5,$load15,$cpu_user,$cpu_system,$cpu_idle" >> "$cpu_mem_log"

    sleep "$MONITOR_INTERVAL"
  done
}

monitor_disk() {
  while true; do
    local ts epoch project_kb fs_line fs_used fs_avail fs_use_pct
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    epoch="$(date +%s)"

    project_kb="$(du -sk "$PROJECT_ROOT" 2>/dev/null | awk '{print $1}')"
    fs_line="$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $3","$4","$5}')"
    fs_used="$(echo "$fs_line" | cut -d',' -f1)"
    fs_avail="$(echo "$fs_line" | cut -d',' -f2)"
    fs_use_pct="$(echo "$fs_line" | cut -d',' -f3)"

    echo "$ts,$epoch,$project_kb,$fs_used,$fs_avail,$fs_use_pct" >> "$disk_log"

    sleep "$MONITOR_INTERVAL"
  done
}

# Inicia monitores
monitor_gpu &
GPU_MON_PID=$!

monitor_cpu_mem &
CPU_MEM_MON_PID=$!

monitor_disk &
DISK_MON_PID=$!

# Executa comando principal
{
  echo "===== START $(date '+%Y-%m-%d %H:%M:%S') ====="
  echo "Run dir: $run_dir"
  echo "Command:"
  printf '%q ' "$@"
  echo
  echo "==============================================="
} | tee -a "$combined_file"

(
  "$@" \
    > >(tee -a "$stdout_file" "$combined_file") \
    2> >(tee -a "$stderr_file" "$combined_file" >&2)
) &
CMD_PID=$!

wait "$CMD_PID"
cmd_exit=$?

end_epoch="$(date +%s)"
end_human="$(date '+%Y-%m-%d %H:%M:%S')"
duration_sec=$(( end_epoch - start_epoch ))

{
  echo "run_dir=$run_dir"
  echo "start_time=$start_human"
  echo "end_time=$end_human"
  echo "duration_seconds=$duration_sec"
  echo "exit_code=$cmd_exit"
} > "$summary_file"

exit "$cmd_exit"
