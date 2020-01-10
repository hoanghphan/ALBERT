#!/bin/bash

HOME="/home/hphan"
ALBERT_ROOT="/home/hphan/data/albert_base"
OUTPUT_DIR="/home/hphan/output"
DATA_DIR="/home/hphan/data/glue"


python -m run_classifier \
  --data_dir=${DATA_DIR} \
  --output_dir=${OUTPUT_DIR} \
  --init_checkpoint=${ALBERT_ROOT} \
  --albert_config_file=${ALBERT_ROOT}/albert_config.json \
  --spm_model_file=${ALBERT_ROOT}/30k-clean.model \
  --do_train=True \
  --do_eval=True \
  --do_predict=False \
  --do_lower_case=True \
  --max_seq_length=128 \
  --optimizer=adamw \
  --task_name=MNLI \
  --warmup_step=1000 \
  --learning_rate=3e-5 \
  --train_step=10000 \
  --save_checkpoints_steps=100 \
  --train_batch_size=128

