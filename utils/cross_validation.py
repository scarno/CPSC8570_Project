"""
cross_validation.py
Updated cross-validation function to evaluate the global model using torch.
"""
import torch
import numpy as np

def validate_model(model, val_loader, dataset_name=None):
    model.model.eval()
    correct = 0
    total = 0
    num_classes = 10
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for x, y in val_loader:
            outputs = model.model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            for i, j in zip(y.cpu().numpy(), predicted.cpu().numpy()):
                confusion[i, j] += 1

    accuracy = correct / total

    print("\nConfusion Matrix:")
    print(confusion)

    per_class_acc = np.diag(confusion) / np.sum(confusion, axis=1)
    print("\nPer-class accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"Class {i}: {acc:.4f}")

    misclass_rate = None
    source_class = None
    target_class = None

    if dataset_name == 'FashionMNIST':
        source_class = 4  # coat
        target_class = 6  # shirt
    elif dataset_name == 'CIFAR10':
        source_class = 0  # bird
        target_class = 2  # airplane

    if source_class is not None and confusion[source_class].sum() > 0:
        misclass_rate = confusion[source_class, target_class] / np.sum(confusion[source_class])

    metrics = {
    'accuracy': accuracy,
    'per_class_recall': per_class_acc,
    'source_class_recall': per_class_acc[source_class] if source_class is not None else None,
    'source_to_target_rate': misclass_rate,
    'source_class_id': source_class,
    'target_class_id': target_class
}

    return metrics
