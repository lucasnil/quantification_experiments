#!/bin/bash

# ❗ ADICIONE O CAMINHO CORRETO AQUI ❗
# Define o caminho para o diretório pai da biblioteca 'dlquantification'
export PYTHONPATH=/home/nildaimon/gmnet

# Define o nome do script Python a ser executado
SCRIPT_PATH="train_miniboone.py"
# Define o diretório onde os resultados (CSVs) serão salvos
RESULT_DIR="resultados_miniboone"

# Cria os diretórios para logs e resultados, se não existirem
mkdir -p logs
mkdir -p $RESULT_DIR

# Loop para executar o experimento 5 vezes (5 runs)
for run in {1..5}; do
  echo "=== Executando MiniBooNE: Rodada $run ==="

  # --- Experimento 1: Sem curriculum learning ---
  python -u $SCRIPT_PATH \
    --run $run \
    --result_path $RESULT_DIR \
    | tee logs/miniboone_none_run${run}.log

  # --- Experimento 2: Métrica KL, modo 'hardest' ---
  python -u $SCRIPT_PATH \
    --difficulty_metric kl \
    --difficulty_mode hardest \
    --run $run \
    --result_path $RESULT_DIR \
    | tee logs/miniboone_kl_hardest_run${run}.log

  # --- Experimento 3: Métrica KL, modo 'easiest' ---
  python -u $SCRIPT_PATH \
    --difficulty_metric kl \
    --difficulty_mode easiest \
    --run $run \
    --result_path $RESULT_DIR \
    | tee logs/miniboone_kl_easiest_run${run}.log

  # --- Experimento 4: Métrica L1, modo 'hardest' ---
  python -u $SCRIPT_PATH \
    --difficulty_metric l1 \
    --difficulty_mode hardest \
    --run $run \
    --result_path $RESULT_DIR \
    | tee logs/miniboone_l1_hardest_run${run}.log

  # --- Experimento 5: Métrica L1, modo 'easiest' ---
  python -u $SCRIPT_PATH \
    --difficulty_metric l1 \
    --difficulty_mode easiest \
    --run $run \
    --result_path $RESULT_DIR \
    | tee logs/miniboone_l1_easiest_run${run}.log
done

echo "=== Todas as execuções do MiniBooNE foram concluídas. ==="