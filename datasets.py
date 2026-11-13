"""Seeded 80/20 splits of official training sets and client label partitions."""

import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class DatasetSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)
        self.targets = np.asarray(dataset.targets)[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, _ = self.dataset[self.indices[index]]
        return image, int(self.targets[index])


# Per-dataset input normalization (mean, std).
NORMALIZATION = {'mnist': ((0.1307,), (0.3081,)), 'fmnist': ((0.1307,), (0.3081,)),
                 'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))}


def partition_clients(targets, num_users, seed, iid=False, alpha=0.5, min_size=1):
    rng = np.random.default_rng(seed)
    if iid:
        return {i: part.tolist() for i, part in enumerate(
            np.array_split(rng.permutation(len(targets)), num_users))}
    targets = np.asarray(targets)
    # Redraw when a client would hold fewer than min_size samples.
    for _ in range(100):
        groups = {i: [] for i in range(num_users)}
        for label in np.unique(targets):
            indices = rng.permutation(np.flatnonzero(targets == label))
            proportions = rng.dirichlet(np.full(num_users, alpha))
            counts = rng.multinomial(len(indices), proportions)
            for i, part in enumerate(np.split(indices, np.cumsum(counts)[:-1])):
                groups[i].extend(part.tolist())
        if min(len(group) for group in groups.values()) >= min_size:
            return groups
    raise ValueError('Dirichlet partition left a client below min_size after 100 draws')


def load_data(args):
    source_class = {'mnist': datasets.MNIST, 'fmnist': datasets.FashionMNIST,
                    'cifar10': datasets.CIFAR10}[args.dataset]
    # Pixel scaling and normalization only: no augmentation or separate official test set.
    mean, std = NORMALIZATION[args.dataset]
    source = source_class(args.data_dir, train=True, download=True,
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize(mean, std)]))
    order = np.random.default_rng(args.seed).permutation(len(source))
    boundary = int(0.8 * len(source))
    train, test = DatasetSubset(source, order[:boundary]), DatasetSubset(source, order[boundary:])
    groups = partition_clients(train.targets, args.num_users, args.seed, args.iid, args.alpha,
                               getattr(args, 'dirichlet_min_size', 1))
    return train, test, groups
