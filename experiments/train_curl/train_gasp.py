import argparse
import os
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from dlquantification.gmnet import GMNet
from dlquantification.featureextraction.nofe import NoFeatureExtractionModule
from dlquantification.utils.lossfunc import MAE
from dlquantification.utils.utils import UnlabeledBagGenerator
from dlquantification.utils.lequabaggenerator import LeQuaBagGenerator


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


def main(dataset_path, difficulty_metric, difficulty_top_k, difficulty_mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo de treino:", device)

    # Configurações fixas
    SEED = 42
    BAG_SIZE = 100
    N_CLASSES = 3
    EMBEDDING_SIZE = 1  # Usando apenas o preço como feature
    TRAIN_NAME = "gasp_gmnet"

    print("Carregando e processando dados GASP")
    df = pd.read_csv(dataset_path)

    # Discretiza o sentimento em 3 classes com quantiles
    est = KBinsDiscretizer(n_bins=N_CLASSES, encode='ordinal', strategy='quantile')
    y = est.fit_transform(df[['Sentiment']]).astype(int).flatten()

    # Usa apenas o preço como feature
    x = df[['Price']].values.astype(np.float32)
    embeddings = torch.from_numpy(x)
    labels = torch.from_numpy(y)

    # Divisão dos dados
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
    y_val = torch.from_numpy(y_val_np)
    y_test = torch.from_numpy(y_test_np)

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

    train_bag_generator = LeQuaBagGenerator(
        device='cpu',
        seed=SEED,
        prevalences=train_prevalences,
        sample_size=BAG_SIZE,
        app_bags_proportion=0.5,
        mixed_bags_proportion=0.5,
        labeled_unlabeled_split=(range(0, n_labeled), range(n_labeled, 2 * n_labeled)),
        difficulty_metric=difficulty_metric,
        difficulty_top_k=difficulty_top_k,
        difficulty_mode=difficulty_mode
    )

    val_bag_generator = UnlabeledBagGenerator(
        device='cpu',
        pick_all=False,
        seed=SEED,
        prevalences=val_prevalences,
        sample_size=BAG_SIZE
    )

    test_bag_generator = UnlabeledBagGenerator(
        device='cpu',
        pick_all=False,
        seed=SEED,
        prevalences=test_prevalences,
        sample_size=BAG_SIZE
    )


    fe = NoFeatureExtractionModule(input_size=EMBEDDING_SIZE)
    loss = MAE()

    params = {
        "n_classes": N_CLASSES,
        "random_seed": SEED,
        "feature_extraction_module": fe,
        "bag_generator": train_bag_generator,
        "val_bag_generator": val_bag_generator,
        "test_bag_generator": test_bag_generator,
        "device": device,
        "quant_loss": loss,
        "dataset_name": "GASP",
        "save_model_path": f"savedmodels/{TRAIN_NAME}.pkl",
        "wandb_experiment_name": TRAIN_NAME,
        "use_wandb": False,
        "use_multiple_devices": False,
        "num_workers": 4,
        "train_epochs": 1,
        "test_epochs": 1,
        "start_lr": 1e-3,
        "end_lr": 1e-5,
        "lr_factor": 0.5,
        "batch_size": 128,
        "gradient_accumulation": 1,
        "weight_decay": 0.001,
        "dropout": 0.5,
        "cka_regularization": "view",
        "n_bags": [5000, 300, 1],
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

    # antes

    # --- após model.fit(...) ---

    # 1) Reconstrói o dataset de bags (não achatado):
    test_dataset_bags = torch.utils.data.TensorDataset(x_test_bags)  
    # cada elemento do dataset é um tensor de shape (BAG_SIZE, EMBEDDING_SIZE)

    # 2) Número de bags de teste:
    n_bags = x_test_bags.shape[0]

    # 3) Chama o predict em modo “process_in_batches” para retornar todas as bags:
    #    (isso corresponde ao “Case 3” na sua implementação)
    preds_bags_np = model.predict(
        test_dataset_bags,
        process_in_batches=n_bags     # força predição uma bag por vez
    )
    # preds_bags_np: numpy array de shape (n_bags, N_CLASSES)

    # 4) Converte para tensor e calcula MAE por bag:
    preds_bags = preds_bags_np.to(test_prevalences.dtype)

    mae_per_bag = torch.nn.functional.l1_loss(
        preds_bags,                     # (n_bags, 3)
        test_prevalences,               # (n_bags, 3)
        reduction="none"
    ).mean(dim=1)                       # calcula MAE ao longo das classes para cada bag

    print(f"MAE médio por bag: {mae_per_bag.mean().item():.4f}")

    df_mae = pd.DataFrame({"bag_id": list(range(len(mae_per_bag))),"mae": mae_per_bag.cpu().numpy()})

    df_mae.to_csv("mae_por_bag.csv", index=False)
    print("Salvo em mae_por_bag.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Caminho para o arquivo .csv do GASP")
    parser.add_argument("--difficulty_metric", default=None, choices=["l1", "kl"], help="Métrica de dificuldade")
    parser.add_argument("--difficulty_top_k", type=int, default=None, help="Top-K instâncias mais difíceis")
    parser.add_argument("--difficulty_mode", default=None, choices=["hardest", "easiest"], help="Modo de dificuldade")

    args = parser.parse_args()

    main(
        dataset_path=args.data,
        difficulty_metric=args.difficulty_metric,
        difficulty_top_k=args.difficulty_top_k,
        difficulty_mode=args.difficulty_mode
    )
