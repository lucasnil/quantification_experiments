#!/bin/bash

echo "--- [$(date)] INICIANDO SCRIPT MESTRE DE EXECUÇÃO DE TODOS OS DATASETS ---"

# ❗ AJUSTE O CAMINHO AQUI ❗
# Garante que o Python encontre a biblioteca 'dlquantification' para todos os sub-scripts.
export PYTHONPATH=/home/nildaimon/gmnet

# Cria o diretório de logs se ele não existir
mkdir -p logs

# --- ETAPA 1: SANDERS ---
echo "--- [$(date)] Iniciando a suíte de experimentos 'Sanders' ---"
start_time=$(date +%s)
bash run_all_sanders_experiments.sh > logs/sanders_runner.log 2>&1
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "--- [$(date)] Suíte 'Sanders' concluída em ${elapsed}s. Log salvo em logs/sanders_runner.log ---"

# --- ETAPA 2: YELP ---
echo "--- [$(date)] Iniciando a suíte de experimentos 'Yelp' ---"
start_time=$(date +%s)
bash run_all_yelp_experiments.sh > logs/yelp_runner.log 2>&1
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "--- [$(date)] Suíte 'Yelp' concluída em ${elapsed}s. Log salvo em logs/yelp_runner.log ---"

# --- ETAPA 3: TWEET EVAL ---
echo "--- [$(date)] Iniciando a suíte de experimentos 'TweetEval' ---"
start_time=$(date +%s)
bash run_all_tweet_eval_experiments.sh > logs/tweet_eval_runner.log 2>&1
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "--- [$(date)] Suíte 'TweetEval' concluída em ${elapsed}s. Log salvo em logs/tweet_eval_runner.log ---"

# --- ETAPA 4: FINANCIAL PHRASEBANK ---
echo "--- [$(date)] Iniciando a suíte de experimentos 'FinancialPhraseBank' ---"
start_time=$(date +%s)
bash run_all_financial_experiments.sh > logs/financial_runner.log 2>&1
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "--- [$(date)] Suíte 'FinancialPhraseBank' concluída em ${elapsed}s. Log salvo em logs/financial_runner.log ---"

# --- ETAPA 5: SST-5 ---
echo "--- [$(date)] Iniciando a suíte de experimentos 'SST-5' ---"
start_time=$(date +%s)
bash run_all_sst_experiments.sh > logs/sst_runner.log 2>&1
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "--- [$(date)] Suíte 'SST-5' concluída em ${elapsed}s. Log salvo em logs/sst_runner.log ---"


echo "--- [$(date)] SCRIPT MESTRE CONCLUÍDO. TODOS OS EXPERIMENTOS FORAM FINALIZADOS. ---"