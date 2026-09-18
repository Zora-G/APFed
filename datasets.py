import json
from pathlib import Path

from torchvision import datasets, transforms


class DatasetSubset(object):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)
        targets = dataset.targets
        try:
            self.targets = targets[self.indices]
        except TypeError:
            self.targets = [targets[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        image, _ = self.dataset[self.indices[item]]
        return image, self.targets[item]


def load_data(data_dir, inputs_path):
    inputs = json.loads(Path(inputs_path).read_text())
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    source = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    train = DatasetSubset(source, inputs['train_source_indices'])
    evaluation = DatasetSubset(source, inputs['test_source_indices'])
    groups = {int(client): indices for client, indices in inputs['user_groups'].items()}
    return train, evaluation, groups, inputs['training_seeds']
