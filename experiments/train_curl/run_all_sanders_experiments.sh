#!/bin/bash
export PYTHONPATH=/home/nildaimon/gmnet

DATA_PATH="/home/nildaimon/gmnet/experiments/train_curl/Datasets/Sanders/twitter-sanders-apple3.csv"
SCRIPT_PATH="train_sanders.py"
RESULT_DIR="resultados_sanders"

mkdir -p logs
mkdir -p $RESULT_DIR



# Sem curriculum
python -u $SCRIPT_PATH \
  --data $DATA_PATH \
  
  --result_path $RESULT_DIR \
  | tee logs/sanders_none_run${run}.log

# KL - hardest
python -u $SCRIPT_PATH \
  --data $DATA_PATH \
  --difficulty_metric kl \
  --difficulty_mode hardest \
  
  --result_path $RESULT_DIR \
  | tee logs/sanders_kl_hardest_run${run}.log

# KL - easiest
python -u $SCRIPT_PATH \
  --data $DATA_PATH \
  --difficulty_metric kl \
  --difficulty_mode easiest \
  
  --result_path $RESULT_DIR \
  | tee logs/sanders_kl_easiest_run${run}.log

# L1 - hardest
python -u $SCRIPT_PATH \
  --data $DATA_PATH \
  --difficulty_metric l1 \
  --difficulty_mode hardest \
  
  --result_path $RESULT_DIR \
  | tee logs/sanders_l1_hardest_run${run}.log

# L1 - easiest
python -u $SCRIPT_PATH \
  --data $DATA_PATH \
  --difficulty_metric l1 \
  --difficulty_mode easiest \
  
  --result_path $RESULT_DIR \
  | tee logs/sanders_l1_easiest_run${run}.log


echo "=== Todas as execuções do Sanders foram concluídas. ==="