#!/bin/bash
export PYTHONPATH=/home/nildaimon/gmnet

SCRIPT_PATH="train_sst.py"
RESULT_DIR="resultados_sst"

mkdir -p logs
mkdir -p $RESULT_DIR

echo "=== Apagando modelos salvos do SST-5 ==="

rm -f savedmodels/sst5_sentiment*

#verifica se apagou
if ls savedmodels/sst5_sentiment* 1> /dev/null 2>&1; then
   echo "Modelos do SST-5 não foram apagados."
else
   echo "Modelos do SST-5 apagados com sucesso."
fi

# Sem curriculum
python -u $SCRIPT_PATH \
  --result_path $RESULT_DIR \
  | tee logs/sst_none_run${run}.log

# KL - hardest
python -u $SCRIPT_PATH \
  --difficulty_metric kl \
  --difficulty_mode hardest \
  --result_path $RESULT_DIR \
  | tee logs/sst_kl_hardest_run${run}.log

# KL - easiest
python -u $SCRIPT_PATH \
  --difficulty_metric kl \
  --difficulty_mode easiest \
  --result_path $RESULT_DIR \
  | tee logs/sst_kl_easiest_run${run}.log

# L1 - hardest
python -u $SCRIPT_PATH \
  --difficulty_metric l1 \
  --difficulty_mode hardest \
  --result_path $RESULT_DIR \
  | tee logs/sst_l1_hardest_run${run}.log

# L1 - easiest
python -u $SCRIPT_PATH \
  --difficulty_metric l1 \
  --difficulty_mode easiest \
  --result_path $RESULT_DIR \
  | tee logs/sst_l1_easiest_run${run}.log

echo "=== Todas as execuções do SST-5 foram concluídas. ==="
