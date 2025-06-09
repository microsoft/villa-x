#!/bin/bash

# better to run jobs for each task
TASK_CONFIGS=(
    "google_robot_pick_horizontal_coke_can:100"
    "google_robot_pick_vertical_coke_can:100"
    "google_robot_pick_standing_coke_can:100"
    "google_robot_move_near_v0:240"
    "google_robot_open_drawer:108"
    "google_robot_close_drawer:108"
    "google_robot_place_apple_in_closed_top_drawer:108"
)
CONFIG_NAME=eval_rt1_simpler

for TASK_CONFIG in "${TASK_CONFIGS[@]}"; do
    TASK="${TASK_CONFIG%%:*}"
    EPISODES="${TASK_CONFIG##*:}"

    echo """MS2_REAL2SIM_ASSET_DIR=./ManiSkill2_real2sim/data \
        VLA_LOG_DIR=./outputs/$CONFIG_NAME/$TASK \
        VLA_DATA_DIR=/path/to/dataset/ \
        VLA_NAME=$CONFIG_NAME \
        TRANSFORMERS_CACHE=/path/to/paligemma-3b-pt-224/ \
        uv run torchrun \
        --standalone \
        scripts/run.py \
        --config-name=$CONFIG_NAME \
        env.task=$TASK \
        use_bf16=False \
        n_eval_episode=$EPISODES \
        'checkpoint_path=/path/to/checkpoint'"""

done
