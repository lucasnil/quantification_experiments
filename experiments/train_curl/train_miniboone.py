import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

# --- FUNÇÃO DE PREPARAÇÃO DE DADOS DO MINIBOONE ---
def prep_data(binned=False):
    """
    Baixa, limpa e prepara o conjunto de dados MiniBooNE.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt"
    print("Baixando e preparando o dataset MiniBooNE...")
    
    colnames = ["att" + str(i + 1) for i in range(50)]
    dta = pd.read_csv(url,
                      header=None,
                      names=colnames,
                      skiprows=1,
                      sep=' ',
                      na_values="-0.999000E+03",
                      skipinitialspace=True)
    
    dta["signal"] = 0
    dta.iloc[:36499, -1] = 1 # 1 para sinal, 0 para fundo
    dta = dta.dropna()

    if binned:
        for col in list(dta)[:-1]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")
            
    print("Dataset MiniBooNE pronto.")
    return dta

# --- CONSTANTES GLOBAIS ---
from dlquantification.gmnet import GMNet
from dlquantification.featureextraction.nofe import NoFeatureExtractionModule
from dlquantification.utils.lossfunc import MAE
from dlquantification.utils.utils import UnlabeledMixerBagGenerator, UnlabeledBagGenerator
from dlquantification.utils.lequabaggenerator import LeQuaBagGenerator
from tqdm import tqdm

SEED = 42
BAG_SIZE = 100
N_CLASSES = 2
EMBEDDING_SIZE = 50
TRAIN_NAME = "miniboone"

torch.manual_seed(SEED)

# --- FUNÇÕES AUXILIARES ---
def padronizar(x_train, x_val, x_test):
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0)
    std[std == 0] = 1e-8
    return (x_train - mean)/std, (x_val - mean)/std, (x_test - mean)/std, mean, std


def criar_bags(x, y, bag_size, n_classes):
    total_samples = len(x)
    n_bags = total_samples // bag_size
    if n_bags == 0:
        print(f"[criar_bags] bag_size={bag_size} é muito grande para o número de amostras ({total_samples}).")
        empty_shape = (0, bag_size, x.shape[1])
        return torch.empty(empty_shape), torch.empty((0, bag_size), dtype=torch.long), torch.empty((0, n_classes))
    # Embaralha os índices antes de truncar
    shuffled_indices = torch.randperm(total_samples)
    x, y = x[shuffled_indices], y[shuffled_indices]

    x = x[:n_bags * bag_size]
    y = y[:n_bags * bag_size]
    x_bags = x.view(n_bags, bag_size, -1)
    y_bags = y.view(n_bags, bag_size)
    prevalences = torch.stack([(y_bags == i).sum(dim=1) / bag_size for i in range(n_classes)], dim=1)
    return x_bags, y_bags, prevalences


# --- FUNÇÃO PRINCIPAL ---
def main(difficulty_metric=None, difficulty_top_k=None, difficulty_mode=None, result_path=".", run=1):
    print(f"Dataset: {TRAIN_NAME}")
    print(f"Dificuldade: metric={difficulty_metric}, mode={difficulty_mode}, top_k={difficulty_top_k}")
    print(f"Rodada: {run}")
    print(f"Salvar resultados em: {result_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo de treino:", device)

    df = prep_data()
    features = torch.tensor(df.drop('signal', axis=1).values, dtype=torch.float32)
    labels = torch.tensor(df['signal'].values, dtype=torch.long)
    
    x_temp, x_test, y_temp, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=SEED
    )

    x_train, x_val, x_test, mean, std = padronizar(x_train, x_val, x_test)

    x_train_bags, y_train_bags, train_prevalences = criar_bags(x_train, y_train, BAG_SIZE, N_CLASSES)
    x_val_bags, y_val_bags, val_prevalences = criar_bags(x_val, y_val, BAG_SIZE, N_CLASSES)
    x_test_bags, y_test_bags, test_prevalences = criar_bags(x_test, y_test, BAG_SIZE, N_CLASSES)
    
    if x_train_bags.numel() == 0:
        print("Não foi possível criar bags de treino. Encerrando.")
        return
        
    n_labeled = x_train_bags.view(-1, EMBEDDING_SIZE).shape[0]

    train_dataset = TensorDataset(
        torch.cat([x_train, x_train_bags.view(-1, EMBEDDING_SIZE)]),
        torch.cat([y_train, y_train_bags.view(-1)])
    )

    val_dataset = TensorDataset(
        x_val_bags.view(-1, EMBEDDING_SIZE),
        torch.full((x_val_bags.view(-1, EMBEDDING_SIZE).shape[0],), fill_value=-1, dtype=torch.long)
    )

    train_bag_generator = LeQuaBagGenerator(
        device='cpu', seed=SEED, prevalences=train_prevalences, sample_size=BAG_SIZE,
        app_bags_proportion=0.5, mixed_bags_proportion=1.0,
        labeled_unlabeled_split=(range(0, n_labeled), range(n_labeled, 2 * n_labeled)),
        difficulty_metric=difficulty_metric, difficulty_top_k=difficulty_top_k, difficulty_mode=difficulty_mode
    )
    val_bag_generator = UnlabeledMixerBagGenerator(
        device='cpu',
        prevalences=val_prevalences,  # Este argumento é mantido, mas não será usado para os bags reais
        sample_size=BAG_SIZE,
        real_bags_proportion=0.0,      # <-- A CHAVE: Garante 100% de bags com prevalências aleatórias
        seed=SEED
    )
    test_bag_generator = UnlabeledMixerBagGenerator(
        device='cpu', prevalences=test_prevalences, sample_size=BAG_SIZE,
        real_bags_proportion=0.0, seed=SEED
    )

    total_test_bags = 1000
    x_test_bags_indexes, test_prevalences = test_bag_generator.compute_bags(n_bags=total_test_bags, bag_size=BAG_SIZE)
    x_test_bags = x_test[x_test_bags_indexes.view(-1)].view(total_test_bags, BAG_SIZE, EMBEDDING_SIZE)

    fe = NoFeatureExtractionModule(input_size=EMBEDDING_SIZE)
    loss = MAE()

    difficulty_suffix = "none" if difficulty_metric is None else f"{difficulty_metric}_{difficulty_mode}"
    save_model_path = f"savedmodels/{TRAIN_NAME}_{difficulty_suffix}_run{run}.pkl"

    params = {
        "n_classes": N_CLASSES, "random_seed": SEED, "feature_extraction_module": fe,
        "bag_generator": train_bag_generator, "val_bag_generator": val_bag_generator,
        "test_bag_generator": test_bag_generator, "device": device, "quant_loss": loss,
        "dataset_name": "MiniBooNE", "save_model_path": save_model_path,
        "wandb_experiment_name": TRAIN_NAME, "use_wandb": False, "use_multiple_devices": False,
        "num_workers": 4, "train_epochs": 1000, "test_epochs": 1,
        "start_lr": 5e-03, "end_lr": 1e-04, "lr_factor": 0.5, "batch_size": 128,
        "gradient_accumulation": 1, "weight_decay": 0.001, "dropout": 0.5,
        "cka_regularization": "view", "n_bags": [5000, 300, 1], "bag_size": BAG_SIZE,
        "linear_sizes": [512], "n_gm_layers": 9, "num_gaussians": [10] * 9,
        "gaussian_dimensions": [5] * 9, "patience": 20, "verbose": 8
    }

    model = GMNet(**params)
    model.fit(dataset=train_dataset, val_dataset=val_dataset)

    test_dataset_bags = torch.utils.data.TensorDataset(x_test_bags)
    preds_bags_np = model.predict(test_dataset_bags, process_in_batches=total_test_bags)
    
    if isinstance(preds_bags_np, np.ndarray):
        preds_bags_np = torch.from_numpy(preds_bags_np)
        
    preds_bags = preds_bags_np.to(test_prevalences.dtype)

    mae_per_bag = torch.nn.functional.l1_loss(preds_bags, test_prevalences, reduction="none").mean(dim=1)
    mean_mae = mae_per_bag.mean().item()
    std_mae = mae_per_bag.std().item()
    print(f"MAE médio por bag: {mean_mae:.4f} ± {std_mae:.4f}")

    # --- INÍCIO: ALTERAÇÃO PARA SALVAR PREDIÇÕES ---
    
    # Cria um dicionário com colunas de prevalência VERDADEIRAS (ground truth)
    true_prevalence_columns = {
        f"true_prev_class_{i}": test_prevalences[:, i].cpu().numpy()
        for i in range(test_prevalences.shape[1])
    }

    # Cria um dicionário com colunas de prevalência PREDITAS pelo modelo
    pred_prevalence_columns = {
        f"pred_prev_class_{i}": preds_bags[:, i].cpu().numpy()
        for i in range(preds_bags.shape[1])
    }

    # Junta tudo em um único DataFrame para salvar
    df_resultados = pd.DataFrame({
        "bag_id": list(range(len(mae_per_bag))),
        "mae": mae_per_bag.cpu().numpy(),
        **true_prevalence_columns,
        **pred_prevalence_columns
    })

    # --- FIM: ALTERAÇÃO PARA SALVAR PREDIÇÕES ---

    difficulty_suffix = ""
    if args.difficulty_metric:
        difficulty_suffix += f"_{args.difficulty_metric}"
    if args.difficulty_mode:
        difficulty_suffix += f"_{args.difficulty_mode}"
    if args.difficulty_top_k is not None:
        difficulty_suffix += f"_top{args.difficulty_top_k}"

    csv_path = f"{args.result_path}/resultados_{TRAIN_NAME}{difficulty_suffix}_{args.run}.csv"
    df_resultados.to_csv(csv_path, index=False)
    print(f"Resultados, incluindo predições, salvos em: {csv_path}")


# --- SEÇÃO DE ARGUMENTOS E EXECUÇÃO ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty_metric", default=None, choices=["l1", "kl"], help="Métrica de dificuldade")
    parser.add_argument("--difficulty_mode", default=None, choices=["hardest", "easiest"], help="Modo de dificuldade")
    parser.add_argument("--difficulty_top_k", type=int, default=None, help="Top-K instâncias mais difíceis")
    parser.add_argument("--result_path", default=".", help="Caminho para salvar os resultados")
    parser.add_argument("--run", type=int, default=1, help="Número da execução para diferenciar os resultados")

    args = parser.parse_args()

    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs("savedmodels", exist_ok=True)

    main(
        difficulty_metric=args.difficulty_metric,
        difficulty_top_k=args.difficulty_top_k,
        difficulty_mode=args.difficulty_mode,
        result_path=args.result_path,
        run=args.run
    )