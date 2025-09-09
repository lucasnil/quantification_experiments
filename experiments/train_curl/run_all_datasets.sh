#!/bin/bash

echo "--- [$(date)] INICIANDO SCRIPT MESTRE DE EXECUÇÃO DE TODOS OS DATASETS ---"

# ❗ AJUSTE O CAMINHO AQUI ❗
# Garante que o Python encontre a biblioteca 'dlquantification' para todos os sub-scripts.
export PYTHONPATH=/home/nildaimon/gmnet

# Cria o diretório de logs se ele não existir
mkdir -p logs

# --- ETAPA 1: SANDERS ---
# echo "--- [$(date)] Iniciando a suíte de experimentos 'Sanders' ---"
# bash run_all_sanders_experiments.sh > logs/sanders_runner.log 2>&1
# echo "--- [$(date)] Suíte 'Sanders' concluída. Log salvo em logs/sanders_runner.log ---"
# # --- ETAPA 2: YELP ---
# echo "--- [$(date)] Iniciando a suíte de experimentos 'Yelp' ---"
# bash run_all_yelp_experiments.sh > logs/yelp_runner.log 2>&1
# echo "--- [$(date)] Suíte 'Yelp' concluída. Log salvo em logs/yelp_runner.log ---"
# --- ETAPA 3: TWEET EVAL ---
echo "--- [$(date)] Iniciando a suíte de experimentos 'TweetEval' ---"
bash run_all_tweet_eval_experiments.sh > logs/tweet_eval_runner.log 2>&1
echo "--- [$(date)] Suíte 'TweetEval' concluída. Log salvo em logs/tweet_eval_runner.log ---"


echo "--- [$(date)] SCRIPT MESTRE CONCLUÍDO. TODOS OS EXPERIMENTOS FORAM FINALIZADOS. ---"