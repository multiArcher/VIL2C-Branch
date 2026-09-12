#!/bin/bash
# VIL2C on YMAPPO. Same SMAC budget as YMAPPO: 8 envs x 400 steps, 10M.

EXPERIMENT_NAME=vil2c_ymappo_5m_vs_6m
CONFIG=vil2c_ymappo
ENV_CONFIG=sc2
MAP_NAME=5m_vs_6m
REPEAT_TIMES=1

function update_hyperparams() {
    declare -n arg_dict=$1
    local iter=$2
    arg_dict["config"]="--config=$CONFIG"
    arg_dict["env_config"]="--env-config=$ENV_CONFIG"
    arg_dict["name"]="name=${EXPERIMENT_NAME}_run$((iter))"
    arg_dict["map_name"]="env_args.map_name=$MAP_NAME"
}

CONDA_ENV_NAME="bcrbc"
if [ -z "$SC2PATH" ]; then
    export SC2PATH="$HOME/StarCraft/StarCraftII"
fi
CUDA_DEVICES=0
WORK_DIR="$HOME/workspace/epymarl_based"
LOG_DIR="$WORK_DIR/log"
PYTHON_SCRIPT="src/main.py"
BUFFER_CPU_ONLY=True
DEVICE=cuda

function update_env_params() {
    declare -n arg_dict=$1
    local iter=$2
    arg_dict["buffer_cpu_only"]="buffer_cpu_only=$BUFFER_CPU_ONLY"
    arg_dict["device"]="device=$DEVICE"
}

timestamp() { date +"%Y-%m-%d_%H-%M-%S"; }
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
cd "$WORK_DIR" || exit
PYTHON_SCRIPT_PATH="$WORK_DIR/$PYTHON_SCRIPT"
mkdir -p "$LOG_DIR"
declare -A args
std_log_path="$LOG_DIR/${EXPERIMENT_NAME}_$(timestamp)_log.log"
err_log_path="$LOG_DIR/${EXPERIMENT_NAME}_$(timestamp)_err.log"

echo "$(timestamp) | INFO     | bash         | Train VIL2C-YMAPPO. Experiment: $EXPERIMENT_NAME" | tee -a "$std_log_path"

run_experiment() {
    update_hyperparams args "$1"
    update_env_params args "$1"
    local pre_args=""
    local post_args=""
    for key in "${!args[@]}"; do
        if [[ "${args[$key]}" == --* ]]; then
            pre_args+="${args[$key]} "
        else
            post_args+="${args[$key]} "
        fi
    done
    cmd="conda run -n ${CONDA_ENV_NAME} --no-capture-output python ${PYTHON_SCRIPT_PATH} ${pre_args}with $post_args"
    echo "$(timestamp) | INFO     | bash         | Command: $cmd" | tee -a "$std_log_path"
    if ! eval "$cmd" 1>> "$std_log_path" 2>> "$err_log_path"; then
        echo "$(timestamp) | FATAL    | bash         | Run Failed, see $err_log_path" | tee -a "$std_log_path" "$err_log_path"
        exit 1
    fi
}

for ((i=1; i<=REPEAT_TIMES; i++)); do
    run_experiment $i
done
echo "$(timestamp) | INFO     | bash         | Train script ends." | tee -a "$std_log_path"
