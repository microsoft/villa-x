#!/bin/bash

# better to run jobs for each task
TASK_CONFIGS=(
    "google_robot_pick_horizontal_coke_can:128"
    "google_robot_pick_vertical_coke_can:128"
    "google_robot_pick_standing_coke_can:128"
    "google_robot_move_near_v0:256"
    "google_robot_open_drawer:128"
    "google_robot_close_drawer:128"
    "google_robot_place_apple_in_closed_top_drawer:128"
    "widowx_carrot_on_plate:256"
    "widowx_put_eggplant_in_basket:256"
    "widowx_spoon_on_towel:256"
    "widowx_stack_cube:256"
)
fp=$1
config_path=$(dirname $fp)
config_name=$(basename $fp)
# see the config file for the number of episodes in each task
TS=$(date '+%y_%m_%d_%H_%M_%S')

for TASK_CONFIG in "${TASK_CONFIGS[@]}"; do
    TASK="${TASK_CONFIG%%:*}"
    BATCH_SIZE=64
    # set bs=16 when drawer in task
    if [[ $TASK == *"drawer"* ]]; then
        BATCH_SIZE=16
    fi
    EPISODES="${TASK_CONFIG##*:}"
    REPEATS=$(expr $EPISODES / $BATCH_SIZE)
    echo "Running task: $TASK with bs=$BATCH_SIZE, rep=$REPEATS"

    MS2_REAL2SIM_ASSET_DIR=./ManiSkill2_real2sim/data \
        WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/22c6d36f-b65a-45d4-b394-9a4e731800ce \
        WANDB_API_KEY=3c09aaf084ee4f79599038d5070a6cbcb5a4b163 \
        VLA_LOG_DIR=./outputs/$config_name/$TS/$TASK \
        VLA_DATA_DIR=/mnt/rush_shared/dataset/ \
        VLA_WANDB_ENTITY=hspk \
        VLA_NAME=$config_name \
        TRANSFORMERS_CACHE=/mnt/rush_shared/dataset_tf_pi/paligemma-3b-pt-224/paligemma-3b-pt-224/ \
        uv run torchrun \
        --standalone \
        run.py \
        --config-path=$config_path \
        --config-name=$config_name \
        env.task=$TASK \
        num_envs=$BATCH_SIZE \
        repeats=$REPEATS \
        use_bf16=False \
        ${@:2}

done
