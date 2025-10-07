# train_gmnet_sst5.py
#
# Stanford Sentiment Treebank (Socher et al. 2013)
# 5 classes: 0=very negative, 1=negative, 2=neutral, 3=positive, 4=very positive
#

from datasets import load_dataset
import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, AutoModel
from dlquantification.gmnet import GMNet
from dlquantification.featureextraction.nofe import NoFeatureExtractionModule
from dlquantification.utils.lossfunc import MAE
from dlquantification.utils.utils import UnlabeledMixerBagGenerator, APPBagGenerator, DifficultySortedAPPBagGenerator
from tqdm import tqdm

SEED = 42
BAG_SIZE = 1000
N_CLASSES = 5
EMBEDDING_SIZE = 768
TRAIN_NAME = "sst5_sentiment"
BERT_MODEL = "bert-base-uncased"

SPLIT_CACHE = Path(f"{TRAIN_NAME}_splits.pt")

torch.manual_seed(SEED)


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
    return (x_train - mean)/std, (x_val - mean)/std, (x_test - mean)/std, mean, std


def criar_bags(x, y, bag_size, n_classes):
    total_samples = len(x)
    n_bags = total_samples // bag_size
    if n_bags == 0:
        print(f"[criar_bags] bag_size={bag_size} é muito grande para o número de amostras ({total_samples}).")
        empty_shape = (0, bag_size, x.shape[1])
        return torch.empty(empty_shape), torch.empty((0, bag_size), dtype=torch.long), torch.empty((0, n_classes))
    shuffled_indices = torch.randperm(total_samples)
    x, y = x[shuffled_indices], y[shuffled_indices]

    x = x[:n_bags * bag_size]
    y = y[:n_bags * bag_size]
    x_bags = x.view(n_bags, bag_size, -1)
    y_bags = y.view(n_bags, bag_size)
    prevalences = torch.stack([(y_bags == i).sum(dim=1) / bag_size for i in range(n_classes)], dim=1)
    return x_bags, y_bags, prevalences


def carregar_sst5():
    print("Baixando Stanford Sentiment Treebank (SST-5) do Hugging Face...")
    ds = load_dataset("SetFit/sst5")

    train_df = ds["train"].to_pandas()[["text", "label"]]
    val_df   = ds["validation"].to_pandas()[["text", "label"]]
    test_df  = ds["test"].to_pandas()[["text", "label"]]

    return train_df, val_df, test_df


def main(difficulty_metric=None, difficulty_top_k=None, difficulty_mode=None, result_path=".", run=1):
    print("Dataset: SST-5")
    print(f"Dificuldade: {difficulty_metric}, modo: {difficulty_mode}, top_k: {difficulty_top_k}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo de treino:", device)

    train_df, val_df, test_df = carregar_sst5()

    if SPLIT_CACHE.exists():
        print("Carregando embeddings e splits do cache...")
        cache = torch.load(SPLIT_CACHE)
        x_train, y_train = cache["x_train"], cache["y_train"]
        x_val, y_val     = cache["x_val"], cache["y_val"]
        x_test, y_test   = cache["x_test"], cache["y_test"]
    else:
        print("Gerando embeddings com BERT...")
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        model_bert = AutoModel.from_pretrained(BERT_MODEL).to(device).eval()

        train_embeddings = gerar_embeddings(train_df["text"].astype(str).tolist(), tokenizer, model_bert, device)
        val_embeddings   = gerar_embeddings(val_df["text"].astype(str).tolist(), tokenizer, model_bert, device)
        test_embeddings  = gerar_embeddings(test_df["text"].astype(str).tolist(), tokenizer, model_bert, device)

        x_train, y_train = train_embeddings, torch.tensor(train_df["label"].values, dtype=torch.long)
        x_val, y_val     = val_embeddings,   torch.tensor(val_df["label"].values, dtype=torch.long)
        x_test, y_test   = test_embeddings,  torch.tensor(test_df["label"].values, dtype=torch.long)

        torch.save({
            "x_train": x_train, "y_train": y_train,
            "x_val": x_val, "y_val": y_val,
            "x_test": x_test, "y_test": y_test
        }, SPLIT_CACHE)

    x_train, x_val, x_test, mean, std = padronizar(x_train, x_val, x_test)

    train_bag_generator = APPBagGenerator(device='cpu', seed=SEED)
    x_train_bags_indexes, train_prevalences = train_bag_generator.compute_bags(
        n_bags=5000, bag_size=BAG_SIZE, y=y_train
    )
    x_train_bags = x_train[x_train_bags_indexes.view(-1)].view(5000, BAG_SIZE, EMBEDDING_SIZE)
    y_train_bags = y_train[x_train_bags_indexes.view(-1)].view(5000, BAG_SIZE)

    val_bag_generator = APPBagGenerator(device='cpu', seed=SEED+1)
    x_val_bags_indexes, val_prevalences = val_bag_generator.compute_bags(
        n_bags=300, bag_size=BAG_SIZE, y=y_val
    )
    x_val_bags = x_val[x_val_bags_indexes.view(-1)].view(300, BAG_SIZE, EMBEDDING_SIZE)

    train_dataset = TensorDataset(
        torch.cat([x_train, x_train_bags.view(-1, EMBEDDING_SIZE)]),
        torch.cat([y_train, y_train_bags.view(-1)])
    )
    val_dataset = TensorDataset(
        x_val_bags.view(-1, EMBEDDING_SIZE),
        torch.full((x_val_bags.view(-1, EMBEDDING_SIZE).shape[0],), fill_value=-1, dtype=torch.long)
    )

    train_bag_generator = DifficultySortedAPPBagGenerator(
        device='cpu',
        seed=SEED,
        difficulty_metric=difficulty_metric,
        difficulty_top_k=difficulty_top_k,
        difficulty_mode=difficulty_mode
    )

    val_bag_generator = UnlabeledMixerBagGenerator(
        device='cpu',
        prevalences=val_prevalences,
        sample_size=BAG_SIZE,
        real_bags_proportion=0.0,
        seed=SEED
    )

    total_test_bags = 1000
    test_bag_generator = APPBagGenerator(device='cpu', seed=SEED)
    x_test_bags_indexes, test_prevalences = test_bag_generator.compute_bags(
        n_bags=total_test_bags, bag_size=BAG_SIZE, y=y_test
    )
    x_test_bags = x_test[x_test_bags_indexes.view(-1)].view(total_test_bags, BAG_SIZE, EMBEDDING_SIZE)

    fe = NoFeatureExtractionModule(input_size=EMBEDDING_SIZE)
    loss = MAE()

    difficulty_str = ""
    if difficulty_metric is not None:
        difficulty_str += f"_{difficulty_metric}"
    if difficulty_mode is not None:
        difficulty_str += f"_{difficulty_mode}"
    if difficulty_top_k is not None:
        difficulty_str += f"_top{difficulty_top_k}"

    params = {
        "n_classes": N_CLASSES,
        "random_seed": SEED,
        "feature_extraction_module": fe,
        "bag_generator": train_bag_generator,
        "val_bag_generator": val_bag_generator,
        "test_bag_generator": test_bag_generator,
        "device": device,
        "quant_loss": loss,
        "dataset_name": "SST-5",
        "save_model_path": f"savedmodels/{TRAIN_NAME}{difficulty_str}_run{run}.pkl",
        "wandb_experiment_name": TRAIN_NAME,
        "use_wandb": False,
        "num_workers": 4,
        "train_epochs": 1000,
        "test_epochs": 1,
        "start_lr": 5e-03,
        "end_lr": 1e-04,
        "batch_size": 128,
        "weight_decay": 0.001,
        "dropout": 0.5,
        "cka_regularization": "view",
        "n_bags": [5000, 300, 1],
        "bag_size": BAG_SIZE,
        "linear_sizes": [4000],
        "n_gm_layers": 9,
        "num_gaussians": [10] * 9,
        "gaussian_dimensions": [5] * 9,
        "patience": 20,
        "verbose": 8
    }

    model = GMNet(**params)
    model.fit(dataset=train_dataset, val_dataset=val_dataset)

    test_dataset_bags = torch.utils.data.TensorDataset(x_test_bags)
    preds_bags_np = model.predict(test_dataset_bags, process_in_batches=total_test_bags)
    preds_bags = preds_bags_np.to(test_prevalences.dtype)

    mae_per_bag = torch.nn.functional.l1_loss(preds_bags, test_prevalences, reduction="none").mean(dim=1)
    print(f"MAE médio: {mae_per_bag.mean().item():.4f} ± {mae_per_bag.std().item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty_metric", default=None, choices=["l1", "kl"])
    parser.add_argument("--difficulty_mode", default=None, choices=["hardest", "easiest"])
    parser.add_argument("--difficulty_top_k", type=int, default=None)
    parser.add_argument("--result_path", default=".", help="Onde salvar resultados")
    parser.add_argument("--run", type=int, default=1)

    args = parser.parse_args()
    os.makedirs(args.result_path, exist_ok=True)

    main(
        difficulty_metric=args.difficulty_metric,
        difficulty_top_k=args.difficulty_top_k,
        difficulty_mode=args.difficulty_mode,
        result_path=args.result_path,
        run=args.run
    )
