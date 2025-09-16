#!/bin/bash
export PYTHONPATH=/home/nildaimon/gmnet

start_time=$(date +%s)

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
time_start=$(date +%s)
python -u $SCRIPT_PATH \
  --result_path $RESULT_DIR \
  | tee logs/sst_none_run${run}.log
time_end=$(date +%s)
echo "Tempo de execução (sem curriculum): $((time_end - time_start)) segundos."

# KL - hardest
time_start=$(date +%s)
python -u $SCRIPT_PATH \
  --difficulty_metric kl \
  --difficulty_mode hardest \
  --result_path $RESULT_DIR \
  | tee logs/sst_kl_hardest_run${run}.log
time_end=$(date +%s)
echo "Tempo de execução (KL hardest): $((time_end - time_start)) segundos."

# KL - easiest
time_start=$(date +%s)
python -u $SCRIPT_PATH \
  --difficulty_metric kl \
  --difficulty_mode easiest \
  --result_path $RESULT_DIR \
  | tee logs/sst_kl_easiest_run${run}.log
time_end=$(date +%s)
echo "Tempo de execução (KL easiest): $((time_end - time_start)) segundos."

# L1 - hardest
time_start=$(date +%s)
python -u $SCRIPT_PATH \
  --difficulty_metric l1 \
  --difficulty_mode hardest \
  --result_path $RESULT_DIR \
  | tee logs/sst_l1_hardest_run${run}.log
time_end=$(date +%s)
echo "Tempo de execução (L1 hardest): $((time_end - time_start)) segundos."

# L1 - easiest
time_start=$(date +%s)
python -u $SCRIPT_PATH \
  --difficulty_metric l1 \
  --difficulty_mode easiest \
  --result_path $RESULT_DIR \
  | tee logs/sst_l1_easiest_run${run}.log
time_end=$(date +%s)
echo "Tempo de execução (L1 easiest): $((time_end - time_start)) segundos."

end_time=$(date +%s)
total_time=$((end_time - start_time))
echo "Tempo total de execução: ${total_time} segundos."

echo "=== Todas as execuções do SST-5 foram concluídas. ==="
