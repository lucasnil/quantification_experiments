import argparse
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from dlquantification.gmnet import GMNet
from dlquantification.featureextraction.nofe import NoFeatureExtractionModule
from dlquantification.utils.lossfunc import MAE
from dlquantification.utils.utils import UnlabeledBagGenerator
from dlquantification.utils.lequabaggenerator import LeQuaBagGenerator


# ---------------------------
# Funções auxiliares
# ---------------------------
def padronizar(x_train, x_val, x_test):
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0)
    return (x_train - mean)/std, (x_val - mean)/std, (x_test - mean)/std, mean, std


def criar_bags(x, y, bag_size, n_classes):
    total_samples = len(x)
    n_bags = total_samples // bag_size

    if n_bags == 0:
        print(
            f"[criar_bags] bag_size={bag_size} é muito grande para o número de amostras ({total_samples}). "
            f"Reduza o bag_size ou aumente o número de exemplos."
        )
        empty_shape = (0, bag_size, x.shape[1])
        return (
            torch.empty(empty_shape),
            torch.empty((0, bag_size), dtype=torch.long),
            torch.empty((0, n_classes))
        )

    x = x[:n_bags * bag_size]
    y = y[:n_bags * bag_size]
    x_bags = x.view(n_bags, bag_size, -1)
    y_bags = y.view(n_bags, bag_size)
    prevalences = torch.stack([(y_bags == i).sum(dim=1) / bag_size for i in range(n_classes)], dim=1)

    return x_bags, y_bags, prevalences


# ---------------------------
# Script principal
# ---------------------------
def main(dataset_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo de treino:", device)

    # --- CONFIGURAÇÕES ---
    SEED = 42
    BAG_SIZE = 100
    N_CLASSES = 4
    EMBEDDING_SIZE = 768
    TRAIN_NAME = "repro_gmnet_full"

    EMBEDDING_FILE = os.path.join(dataset_path, f"{TRAIN_NAME}_embeddings.npy")
    LABEL_FILE = os.path.join(dataset_path, f"{TRAIN_NAME}_labels.npy")

    print("Carregando embeddings e rótulos...")
    embeddings = torch.from_numpy(np.load(EMBEDDING_FILE))
    labels = torch.from_numpy(np.load(LABEL_FILE))

    # --- Separação dos dados ---
    x_temp, x_test, y_temp, y_test = train_test_split(
        embeddings, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=SEED
    )

    x_train, x_val, x_test, mean, std = padronizar(x_train, x_val, x_test)

    x_train_bags, y_train_bags, train_prevalences = criar_bags(x_train, y_train, BAG_SIZE, N_CLASSES)
    x_val_bags, y_val_bags, val_prevalences = criar_bags(x_val, y_val, BAG_SIZE, N_CLASSES)
    x_test_bags, y_test_bags, test_prevalences = criar_bags(x_test, y_test, BAG_SIZE, N_CLASSES)

    n_labeled = x_train_bags.numel() // EMBEDDING_SIZE

    train_dataset = TensorDataset(
        torch.cat([x_train, x_train_bags.view(-1, EMBEDDING_SIZE)]),
        torch.cat([y_train, y_train_bags.view(-1)])
    )

    val_dataset = TensorDataset(
        x_val_bags.view(-1, EMBEDDING_SIZE),
        torch.full((x_val_bags.view(-1, EMBEDDING_SIZE).shape[0],), fill_value=-1, dtype=torch.long)
    )

    # --- Geradores de bags ---
    train_bag_generator = LeQuaBagGenerator(
        device='cpu',
        seed=SEED,
        prevalences=train_prevalences,
        sample_size=BAG_SIZE,
        app_bags_proportion=0.5,
        mixed_bags_proportion=0.5,
        labeled_unlabeled_split=(range(0, n_labeled), range(n_labeled, 2 * n_labeled)),
        difficulty_metric='l1',
        difficulty_top_k=100,  # Mantém os 100 bags mais fáceis
        difficulty_mode="hardest"  # 'hardest' ou 'easiest'
    )

    val_bag_generator = UnlabeledBagGenerator(
        device='cpu',
        pick_all=False,
        seed=SEED,
        prevalences=val_prevalences,
        sample_size=BAG_SIZE
    )

    # --- Inicialização e treino ---
    fe = NoFeatureExtractionModule(input_size=EMBEDDING_SIZE)
    loss = MAE()

    params = {
        "n_classes": N_CLASSES,
        "random_seed": SEED,
        "feature_extraction_module": fe,
        "bag_generator": train_bag_generator,
        "val_bag_generator": val_bag_generator,
        "device": device,
        "quant_loss": loss,
        "dataset_name": "RePro",
        "save_model_path": f"savedmodels/{TRAIN_NAME}.pkl",
        "wandb_experiment_name": TRAIN_NAME,
        "use_wandb": False,
        "use_multiple_devices": False,
        "num_workers": 4,
        "train_epochs": 150,
        "test_epochs": 1,
        "start_lr": 1e-3,
        "end_lr": 1e-5,
        "lr_factor": 0.5,
        "batch_size": 128,
        "gradient_accumulation": 1,
        "weight_decay": 0.001,
        "dropout": 0.5,
        "cka_regularization": "view",
        "n_bags": [1000, 100, 1],
        "bag_size": BAG_SIZE,
        "linear_sizes": [512, 128],
        "n_gm_layers": 4,
        "num_gaussians": [40] * 4,
        "gaussian_dimensions": [6] * 4,
        "patience": 20,
        "verbose": 8
    }

    model = GMNet(**params)
    model.fit(dataset=train_dataset, val_dataset=val_dataset)

    # --- Avaliação ---
    test_dataset = TensorDataset(x_test_bags)
    preds = model.predict(test_dataset)
    mae = torch.nn.functional.l1_loss(preds, test_prevalences, reduction="none").mean(dim=1)
    print(f"MAE final: {mae.mean().item():.4f}")


# ---------------------------
# Entrada via terminal
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Caminho para a pasta com os arquivos _embeddings.npy e _labels.npy")
    args = parser.parse_args()
    main(args.data)
