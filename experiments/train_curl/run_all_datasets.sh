#!/bin/bash

echo "--- [$(date)] INICIANDO SCRIPT MESTRE DE EXECUÇÃO DE TODOS OS DATASETS ---"

# ❗ AJUSTE O CAMINHO AQUI ❗
# Garante que o Python encontre a biblioteca 'dlquantification' para todos os sub-scripts.
export PYTHONPATH=/home/nildaimon/gmnet

# Cria o diretório de logs se ele não existir
mkdir -p logs

# --- ETAPA 1: SANDERS ---
echo "--- [$(date)] Iniciando a suíte de experimentos 'Sanders' ---"
bash run_all_sanders_experiments.sh > logs/sanders_runner.log 2>&1
echo "--- [$(date)] Suíte 'Sanders' concluída. Log salvo em logs/sanders_runner.log ---"

# --- ETAPA 2: MINIBOONE ---
echo "--- [$(date)] Iniciando a suíte de experimentos 'MiniBooNE' ---"
bash run_all_miniboone_experiments.sh > logs/miniboone_runner.log 2>&1
echo "--- [$(date)] Suíte 'MiniBooNE' concluída. Log salvo em logs/miniboone_runner.log ---"

# --- ETAPA 3: BLOGFEEDBACK ---
echo "--- [$(date)] Iniciando a suíte de experimentos 'BlogFeedback' ---"
bash run_all_blog_experiments.sh > logs/blog_runner.log 2>&1
echo "--- [$(date)] Suíte 'BlogFeedback' concluída. Log salvo em logs/blog_runner.log ---"


echo "--- [$(date)] SCRIPT MESTRE CONCLUÍDO. TODOS OS EXPERIMENTOS FORAM FINALIZADOS. ---"