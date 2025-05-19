set -x

export WG_BACKEND=ray


nproc_per_node=8
CONFIG_PATH="/root/paddlejob/workspace/env_run/RL/verl/verl/trainer/config/checklist.yaml"

python3 -m verl.trainer.main_ppo \
    --config_path=$CONFIG_PATH
