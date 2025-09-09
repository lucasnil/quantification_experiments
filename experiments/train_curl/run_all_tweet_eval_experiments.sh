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
python -u $SCRIPT_PATH \
  --result_path $RESULT_DIR \
  | tee logs/tweet_eval_none_run${run}.log

# KL - hardest
python -u $SCRIPT_PATH \
  --difficulty_metric kl \
  --difficulty_mode hardest \
  --result_path $RESULT_DIR \
  | tee logs/tweet_eval_kl_hardest_run${run}.log

# KL - easiest
python -u $SCRIPT_PATH \
  --difficulty_metric kl \
  --difficulty_mode easiest \
  --result_path $RESULT_DIR \
  | tee logs/tweet_eval_kl_easiest_run${run}.log

# L1 - hardest
python -u $SCRIPT_PATH \
  --difficulty_metric l1 \
  --difficulty_mode hardest \
  --result_path $RESULT_DIR \
  | tee logs/tweet_eval_l1_hardest_run${run}.log

# L1 - easiest
python -u $SCRIPT_PATH \
  --difficulty_metric l1 \
  --difficulty_mode easiest \
  --result_path $RESULT_DIR \
  | tee logs/tweet_eval_l1_easiest_run${run}.log

echo "=== Todas as execuções do TweetEval foram concluídas. ==="
