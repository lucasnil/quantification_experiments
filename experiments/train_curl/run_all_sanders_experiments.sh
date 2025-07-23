#!/bin/bash

DATA_PATH="/home/nildaimon/gmnet/experiments/train_curl/Datasets/Sanders/twitter-sanders-apple3.csv"
SCRIPT_PATH="train_sanders.py"
RESULT_DIR="resultados_sanders"

mkdir -p logs
mkdir -p $RESULT_DIR

for run in {1..5}; do
  echo "=== Execução $run ==="

  # Sem curriculum (baseline)
  nohup python -u $SCRIPT_PATH \
    --data $DATA_PATH \
    --run $run \
    --result_path $RESULT_DIR \
    > logs/sanders_none_run${run}.log 2>&1 &

  # KL - hardest
  nohup python -u $SCRIPT_PATH \
    --data $DATA_PATH \
    --difficulty_metric kl \
    --difficulty_mode hardest \
    --run $run \
    --result_path $RESULT_DIR \
    > logs/sanders_kl_hardest_run${run}.log 2>&1 &

  # KL - easiest
  nohup python -u $SCRIPT_PATH \
    --data $DATA_PATH \
    --difficulty_metric kl \
    --difficulty_mode easiest \
    --run $run \
    --result_path $RESULT_DIR \
    > logs/sanders_kl_easiest_run${run}.log 2>&1 &

  # L1 - hardest
  nohup python -u $SCRIPT_PATH \
    --data $DATA_PATH \
    --difficulty_metric l1 \
    --difficulty_mode hardest \
    --run $run \
    --result_path $RESULT_DIR \
    > logs/sanders_l1_hardest_run${run}.log 2>&1 &

  # L1 - easiest
  nohup python -u $SCRIPT_PATH \
    --data $DATA_PATH \
    --difficulty_metric l1 \
    --difficulty_mode easiest \
    --run $run \
    --result_path $RESULT_DIR \
    > logs/sanders_l1_easiest_run${run}.log 2>&1 &

  # Aguarda todos os processos da rodada atual
  wait
done
