"""Evaluation on the held-out part of the official training set."""

import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(model, dataset, device, source_label=1, target_label=7):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    correct, loss, total, source_count, target_predictions = 0, 0.0, 0, 0, 0
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=128, shuffle=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss += criterion(logits, labels).item()
            predicted = logits.argmax(dim=1)
            correct += int((predicted == labels).sum())
            total += len(labels)
            source = labels == source_label
            source_count += int(source.sum())
            target_predictions += int(((predicted == target_label) & source).sum())
    return {'accuracy': 100.0 * correct / total, 'test_loss': loss / total,
            'attack_success_rate': target_predictions / source_count if source_count else None}
