#!/bin/bash
export PYTHONPATH=/home/nildaimon/gmnet

SCRIPT_PATH="train_tweet_eval.py"
RESULT_DIR="resultados_tweet_eval"

mkdir -p logs
mkdir -p $RESULT_DIR

echo "=== Apagando modelos salvos do TweetEval ==="

rm -f savedmodels/tweet_eval*

#verifica se apagou
if ls savedmodels/tweet_eval* 1> /dev/null 2>&1; then
   echo "Modelos do TweetEval não foram apagados."
else
   echo "Modelos do TweetEval apagados com sucesso."
fi

# Sem curriculum
time_start=$(date +%s)
python -u $SCRIPT_PATH \
  --result_path $RESULT_DIR \
  | tee logs/tweet_eval_none_run${run}.log
time_end=$(date +%s)
echo "Tempo de execução (sem curriculum): $((time_end - time_start)) segundos."

# KL - hardest
time_start=$(date +%s)
python -u $SCRIPT_PATH \
  --difficulty_metric kl \
  --difficulty_mode hardest \
  --result_path $RESULT_DIR \
  | tee logs/tweet_eval_kl_hardest_run${run}.log
time_end=$(date +%s)
echo "Tempo de execução (KL hardest): $((time_end - time_start)) segundos."

# KL - easiest
time_start=$(date +%s)
python -u $SCRIPT_PATH \
  --difficulty_metric kl \
  --difficulty_mode easiest \
  --result_path $RESULT_DIR \
  | tee logs/tweet_eval_kl_easiest_run${run}.log
time_end=$(date +%s)
echo "Tempo de execução (KL easiest): $((time_end - time_start)) segundos."

# L1 - hardest
time_start=$(date +%s)
python -u $SCRIPT_PATH \
  --difficulty_metric l1 \
  --difficulty_mode hardest \
  --result_path $RESULT_DIR \
  | tee logs/tweet_eval_l1_hardest_run${run}.log
time_end=$(date +%s)
echo "Tempo de execução (L1 hardest): $((time_end - time_start)) segundos."

# L1 - easiest
time_start=$(date +%s)
python -u $SCRIPT_PATH \
  --difficulty_metric l1 \
  --difficulty_mode easiest \
  --result_path $RESULT_DIR \
  | tee logs/tweet_eval_l1_easiest_run${run}.log
time_end=$(date +%s)
echo "Tempo de execução (L1 easiest): $((time_end - time_start)) segundos."

echo "=== Todas as execuções do TweetEval foram concluídas. ==="
