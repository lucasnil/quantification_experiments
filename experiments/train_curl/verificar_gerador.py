import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Adicione esta seção para o PYTHONPATH
import sys
project_root = Path(__file__).resolve().parent.parent 
sys.path.append(str(project_root))

# --- MUDANÇA 1: Adicionado novo import ---
from dlquantification.utils.lequabaggenerator import LeQuaBagGenerator
from dlquantification.utils.utils import UnlabeledMixerBagGenerator

# --- CONFIGURAÇÕES GLOBAIS ---
SEED = 42
BAG_SIZE = 100
N_CLASSES = 3
EMBEDDING_SIZE = 768
TRAIN_NAME = "twitter_bert"
BERT_MODEL = "bert-base-uncased"

EMBEDDING_NAME = Path(f"{TRAIN_NAME}_embeddings.pt")
LABELS_NAME = Path(f"{TRAIN_NAME}_labels.pt")

torch.manual_seed(SEED)
np.set_printoptions(precision=3, suppress=True)

# --- FUNÇÕES AUXILIARES (sem alterações) ---
def gerar_embeddings(texts, tokenizer, model, device, batch_size=32, max_length=128):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Gerando embeddings"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1)
            sum_embeddings = (outputs.last_hidden_state * mask).sum(1)
            lengths = mask.sum(1)
            mean_pooled = sum_embeddings / lengths
        embeddings.append(mean_pooled.cpu())
    return torch.cat(embeddings)

def padronizar(x_train, x_val, x_test):
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0)
    std[std == 0] = 1.0
    return (x_train - mean)/std, (x_val - mean)/std, (x_test - mean)/std, mean, std

def criar_bags(x, y, bag_size, n_classes):
    total_samples = len(x)
    n_bags = total_samples // bag_size
    if n_bags == 0:
        return torch.empty((0, bag_size, x.shape[1])), torch.empty((0, n_classes))
    shuffled_indices = torch.randperm(total_samples)
    x, y = x[shuffled_indices], y[shuffled_indices]

    x = x[:n_bags * bag_size]
    y = y[:n_bags * bag_size]
    x_bags = x.view(n_bags, bag_size, -1)
    y_bags = y.view(n_bags, bag_size)
    prevalences = torch.stack([(y_bags == i).sum(dim=1).float() / bag_size for i in range(n_classes)], dim=1)
    return x_bags, prevalences

# --- LÓGICA PRINCIPAL DE VERIFICAÇÃO ---
def main(dataset_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    # ... (carregamento, processamento, divisão de dados - sem alterações) ...
    print("Carregando e processando dataset...")
    df = pd.read_csv(dataset_path)
    label_map = {'Neg': 0, 'Neutral': 1, 'Pos': 2}
    df['label'] = df['class'].map(label_map)

    EMBEDDING_CACHE = Path(dataset_path).parent / EMBEDDING_NAME
    LABELS_CACHE = Path(dataset_path).parent / LABELS_NAME

    if EMBEDDING_CACHE.exists() and LABELS_CACHE.exists():
        embeddings = torch.load(EMBEDDING_CACHE)
        labels = torch.load(LABELS_CACHE)
    else:
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        model_bert = AutoModel.from_pretrained(BERT_MODEL).to(device).eval()
        embeddings = gerar_embeddings(df['text'].astype(str).tolist(), tokenizer, model_bert, device)
        labels = torch.tensor(df['label'].values, dtype=torch.long)
        torch.save(embeddings, EMBEDDING_CACHE)
        torch.save(labels, LABELS_CACHE)

    x_temp_np, x_test_np, y_temp_np, y_test_np = train_test_split(
        embeddings.numpy(), labels.numpy(), test_size=0.2, stratify=labels.numpy(), random_state=SEED
    )
    x_train_np, x_val_np, y_train_np, y_val_np = train_test_split(
        x_temp_np, y_temp_np, test_size=0.25, stratify=y_temp_np, random_state=SEED
    )

    x_train = torch.from_numpy(x_train_np)
    x_val = torch.from_numpy(x_val_np)
    x_test = torch.from_numpy(x_test_np)
    y_train = torch.from_numpy(y_train_np)

    x_train, _, _, _, _ = padronizar(x_train, x_val, x_test)
    x_train_bags, train_prevalences = criar_bags(x_train, y_train, BAG_SIZE, N_CLASSES)
    n_labeled = x_train_bags.numel() // EMBEDDING_SIZE

    # --- Verificação do LeQuaBagGenerator ---
    train_bag_generator = LeQuaBagGenerator(
        device='cpu', seed=SEED, prevalences=train_prevalences, sample_size=BAG_SIZE,
        app_bags_proportion=1.0, mixed_bags_proportion=0,
        labeled_unlabeled_split=(range(0, n_labeled), range(n_labeled, 2 * n_labeled))
    )
    print("\n" + "="*50)
    print("VERIFICANDO O LeQuaBagGenerator")
    print("="*50)
    n_bags_para_verificar = 10
    _, prevalencias_lequa = train_bag_generator.compute_bags(
        y=y_train, n_bags=n_bags_para_verificar, bag_size=BAG_SIZE
    )
    print("\nPrevalências GERADAS pelo LeQuaBagGenerator (somente de APP):")
    print(prevalencias_lequa.numpy())


    # --- MUDANÇA 2: Adicionado novo bloco de verificação para o Bag Mixer ---
    print("\n" + "#"*50)
    print("VERIFICANDO O UnlabeledMixerBagGenerator (Bag Mixer puro)")
    print("#"*50)

    # Inicializa o Bag Mixer para gerar 100% de bags misturados
    bag_mixer_generator = UnlabeledMixerBagGenerator(
        device='cpu',
        prevalences=train_prevalences, # Usa as mesmas prevalências originais como base
        sample_size=BAG_SIZE,
        real_bags_proportion=0.0, # Garante que todos os bags gerados sejam misturados
        seed=SEED
    )

    print(f"Gerando {n_bags_para_verificar} bags de exemplo com o Bag Mixer...")
    
    # Este gerador não precisa de 'y', pois apenas mistura as prevalências existentes
    _, prevalencias_bag_mixer = bag_mixer_generator.compute_bags(
        n_bags=n_bags_para_verificar,
        bag_size=BAG_SIZE
    )

    print("\nPrevalências GERADAS pelo Bag Mixer:")
    print(prevalencias_bag_mixer.numpy())
    
    print("\nLembre-se: cada linha acima deve ser a média de duas linhas das prevalências ORIGINAIS.")
    print("#"*50)
    
    # --- Impressão final de comparação ---
    print("\n" + "*"*50)
    print("COMPARAÇÃO FINAL")
    print("*"*50)
    print("\nPrevalências ORIGINAIS (amostra):")
    print(train_prevalences[:n_bags_para_verificar].numpy())
    print("\nPrevalências do LeQuaBagGenerator (APP + Misto):")
    print(prevalencias_lequa.numpy())
    print("\nPrevalências do Bag Mixer (Apenas Misto):")
    print(prevalencias_bag_mixer.numpy())
    print("\nVerificação concluída.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifica e compara diferentes geradores de bags.")
    parser.add_argument("--data", required=True, help="Caminho para o arquivo .csv do dataset.")
    args = parser.parse_args()
    main(dataset_path=args.data)