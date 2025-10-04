#!/bin/bash
export PYTHONPATH=/home/nildaimon/gmnet

start_time=$(date +%s)

DATA_PATH="yelp"
SCRIPT_PATH="train_yelp.py"
RESULT_DIR="resultados_yelp"

mkdir -p logs
mkdir -p $RESULT_DIR

echo "=== Apagando modelos salvos do Yelp ==="

rm -f savedmodels/yelp*

#verifica se apagou
if ls savedmodels/yelp* 1> /dev/null 2>&1; then
   echo "Modelos do Yelp não foram apagados."
else
   echo "Modelos do Yelp apagados com sucesso."
fi

# Sem curriculum
time_start=$(date +%s)
python -u $SCRIPT_PATH \
  --data $DATA_PATH \
  --result_path $RESULT_DIR \
  | tee logs/yelp_none_run${run}.log
time_end=$(date +%s)
echo "Tempo de execução (sem curriculum): $((time_end - time_start)) segundos."

# KL - hardest
time_start=$(date +%s)
python -u $SCRIPT_PATH \
  --data $DATA_PATH \
  --difficulty_metric kl \
  --difficulty_mode hardest \
  --difficulty_top_k 50 \
  --result_path $RESULT_DIR \
  | tee logs/yelp_kl_hardest_run${run}.log
time_end=$(date +%s)
echo "Tempo de execução (KL hardest): $((time_end - time_start)) segundos."

# KL - easiest
time_start=$(date +%s)
python -u $SCRIPT_PATH \
  --data $DATA_PATH \
  --difficulty_metric kl \
  --difficulty_mode easiest \
  --difficulty_top_k 50 \
  --result_path $RESULT_DIR \
  | tee logs/yelp_kl_easiest_run${run}.log
time_end=$(date +%s)
echo "Tempo de execução (KL easiest): $((time_end - time_start)) segundos."

# L1 - hardest
time_start=$(date +%s)
python -u $SCRIPT_PATH \
  --data $DATA_PATH \
  --difficulty_metric l1 \
  --difficulty_mode hardest \
  --difficulty_top_k 50 \
  --result_path $RESULT_DIR \
  | tee logs/yelp_l1_hardest_run${run}.log
time_end=$(date +%s)
echo "Tempo de execução (L1 hardest): $((time_end - time_start)) segundos."

# L1 - easiest
time_start=$(date +%s)
python -u $SCRIPT_PATH \
  --data $DATA_PATH \
  --difficulty_metric l1 \
  --difficulty_mode easiest \
  --difficulty_top_k 50 \
  --result_path $RESULT_DIR \
  | tee logs/yelp_l1_easiest_run${run}.log
time_end=$(date +%s)
echo "Tempo de execução (L1 easiest): $((time_end - time_start)) segundos."

end_time=$(date +%s)
total_time=$((end_time - start_time))
echo "Tempo total de execução: ${total_time} segundos."

echo "=== Todas as execuções do Yelp foram concluídas. ==="
