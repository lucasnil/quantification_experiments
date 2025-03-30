from dlquantification.utils.utils import BaseBagGenerator, UnlabeledMixerBagGenerator, APPBagGenerator
import torch
import torch.nn.functional as F

class LeQuaBagGenerator(BaseBagGenerator):
    def __init__(
        self,
        device,
        seed,
        prevalences,
        sample_size,
        app_bags_proportion,
        mixed_bags_proportion,
        labeled_unlabeled_split,
        difficulty_metric=None  # Novo parâmetro para escolher a métrica de dificuldade ('l1' ou 'kl')
    ):
        self.device = device
        self.appBagGenerator = APPBagGenerator(device=device, seed=seed)
        self.unlabeledMixerBagGenerator = UnlabeledMixerBagGenerator(
            device=device,
            prevalences=prevalences,
            sample_size=sample_size,
            real_bags_proportion=1 - mixed_bags_proportion,
            seed=seed,
        )
        self.labeled_unlabeled_split = labeled_unlabeled_split
        self.mixed_bags_proportion = mixed_bags_proportion
        self.app_bags_proportion = app_bags_proportion
        self.seed = seed
        self.sample_size = sample_size
        self.labeled_indexes = labeled_unlabeled_split[0]
        self.unlabeled_indexes = labeled_unlabeled_split[1]
        self.difficulty_metric = difficulty_metric  # Armazena a métrica de dificuldade selecionada

    def compute_bags(self, n_bags: int, bag_size: int, y):
        app_bags = round(n_bags * self.app_bags_proportion)
        bags_from_unlabeled = n_bags - app_bags

        samples_app_indexes, prevalences_app = self.appBagGenerator.compute_bags(
            n_bags=app_bags, bag_size=bag_size, y=y[self.labeled_indexes]
        )
        if bags_from_unlabeled > 0:
            samples_unlabeled_indexes, prevalences_unlabeled = self.unlabeledMixerBagGenerator.compute_bags(
                n_bags=bags_from_unlabeled, bag_size=bag_size
            )
            # Corrige os índices para os dados não rotulados (começam em len(labeled_indexes))
            samples_unlabeled_indexes = torch.add(samples_unlabeled_indexes, len(self.labeled_indexes))
            samples_indexes = torch.cat((samples_app_indexes, samples_unlabeled_indexes))
            prevalences = torch.cat((prevalences_app, prevalences_unlabeled))
        else:
            samples_indexes = samples_app_indexes
            prevalences = prevalences_app

        # Mistura os bags
        shuffle = torch.randperm(n_bags)
        samples_indexes = samples_indexes[shuffle, :]
        prevalences = prevalences[shuffle, :]

        # Se uma métrica de dificuldade foi definida, ordena os bags
        if self.difficulty_metric is not None:
            # Vetor balanceado: para cada uma das classes, valor ideal é 1/N
            n_classes = prevalences.shape[1]
            balanced = torch.full((n_classes,), 1 / n_classes, device=self.device, dtype=prevalences.dtype)

            if self.difficulty_metric == 'l1':
                # Diferença L1: soma das diferenças absolutas entre cada prevalência e o valor ideal
                difficulty = torch.sum(torch.abs(prevalences - balanced), dim=1)
            elif self.difficulty_metric == 'kl':
                eps = 1e-8
                # Garante que os vetores estejam normalizados (soma = 1); normalmente já são
                # Calcula a divergência KL: sum( p * log(p/q) )
                difficulty = torch.sum(prevalences * torch.log((prevalences + eps) / (balanced + eps)), dim=1)
            else:
                raise ValueError(f"Unknown difficulty_metric: {self.difficulty_metric}")

            # Ordena os bags com base na dificuldade (do menor para o maior, ou seja, os mais "fáceis" primeiro)
            sorted_idxs = torch.argsort(difficulty)
            samples_indexes = samples_indexes[sorted_idxs, :]
            prevalences = prevalences[sorted_idxs, :]

        return samples_indexes, prevalences

    def get_parameters_to_log(self):
        return {'baggeneratorname': 'LeQuaBagGenerator',
                'app_bags_proportion': self.app_bags_proportion,
                'mixed_bags_proportion': self.mixed_bags_proportion,
                'seed': self.seed,
                'sample_size': self.sample_size,
                'difficulty_metric': self.difficulty_metric}
