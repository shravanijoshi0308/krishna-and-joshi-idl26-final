"""This pretraining.py is used for performing transfer learning as part of assignment task."""

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders    
from fit import Trainer
import models

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")   
print(f"Training executing on device: {device}")

train_loader, val_loader, test_loader = get_loaders(data="organs", data_path='Data', batch_size=32)
print(f"Size of training dataset: {len(train_loader.dataset)}")
print(f"Size of validation dataset: {len(val_loader.dataset)}")
print(f"Size of test dataset: {len(test_loader.dataset)}")

print("\n" + "="*50)
print("EXPERIMENT 1: Training from scratch")
print("="*50)
AlexNet_model_from_scratch = models.AlexNet(in_channels=1, num_classes=11, drop_rate=0.5, activation_str="ReLU").to(device)
criterion = nn.CrossEntropyLoss()
optimizer_scratch = optim.Adam(AlexNet_model_from_scratch.parameters(), lr=0.0001)

trainer_scratch = Trainer(AlexNet_model_from_scratch, criterion, optimizer_scratch, device)
trainer_scratch.fit(train_loader, val_loader, epochs=10)

test_loss, test_acc, test_precision, test_recall, test_f1 = trainer_scratch.evaluate(test_loader)
print(f"Scratch Test Results | Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")

print("\n" + "="*50)
print("EXPERIMENT 2: Transfer learning from orgs-trained model")
print("="*50)
model_transfer = models.AlexNet(in_channels=1, num_classes=11, drop_rate=0.5, activation_str="ReLU").to(device)
model_transfer.load_state_dict(torch.load("AlexNet_orgs.pth", map_location=device))
model_transfer.requires_grad_(False)
for param in model_transfer.classifier[-1].parameters():
    param.requires_grad = True

criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.Adam(filter(lambda p: p.requires_grad, model_transfer.parameters()), lr=0.0001)

trainer_transfer = Trainer(model_transfer, criterion_transfer, optimizer_transfer, device)
trainer_transfer.fit(train_loader, val_loader, epochs=10)

test_loss_t, test_acc_t, test_precision_t, test_recall_t, test_f1_t = trainer_transfer.evaluate(test_loader)
print(f"Transfer Test Results | Loss: {test_loss_t:.4f} | Acc: {test_acc_t:.2f}% | Precision: {test_precision_t:.4f} | Recall: {test_recall_t:.4f} | F1: {test_f1_t:.4f}")

# Track results in a structured matrix, matching the runner pattern from Parts 1 & 2
results = []
results.append({
    "experiment": "Scratch",
    "test_loss": test_loss,
    "test_acc": test_acc,
    "precision": test_precision,
    "recall": test_recall,
    "f1": test_f1
})
results.append({
    "experiment": "Transfer (frozen backbone)",
    "test_loss": test_loss_t,
    "test_acc": test_acc_t,
    "precision": test_precision_t,
    "recall": test_recall_t,
    "f1": test_f1_t
})

print("\n" + "="*50)
print("SCARCE-DATA BENCHMARK MATRIX (organs dataset) for AlexNet")
print("="*50)
for r in results:
    print(f"{r['experiment']:<28} | Loss: {r['test_loss']:.4f} | Test Acc: {r['test_acc']:6.2f}% | "
          f"Precision: {r['precision']:.4f} | Recall: {r['recall']:.4f} | F1: {r['f1']:.4f}")

improvement = results[1]['test_acc'] - results[0]['test_acc']
print(f"\nImprovement from transfer learning: +{improvement:.2f}% accuracy")
print(f"Both experiments {'meet' if all(r['test_acc'] >= 40 for r in results) else 'do NOT meet'} the 40% minimum target.")