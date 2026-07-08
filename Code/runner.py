"""This file runs all model/dataset combinations for the official benchmark."""

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from fit import Trainer
import time 

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" 
                      if torch.cuda.is_available() else "cpu"
                      )

DATASET_CONFIG = {
    "cells":   {"CHANNELS": 3, "NUM_CLASSES": 8, "DROPOUT_RATE": 0.5},
    "chest":   {"CHANNELS": 1, "NUM_CLASSES": 2, "DROPOUT_RATE": 0.6},
    "lesions": {"CHANNELS": 3, "NUM_CLASSES": 7, "DROPOUT_RATE": 0.5},
    "orgs":    {"CHANNELS": 1, "NUM_CLASSES": 11, "DROPOUT_RATE": 0.5},
}

MODEL_SET = ["AlexNet", "VGG16", "ResNet18", "GreenNet"]
 
DATA_PATH = "Data"
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 10
ACTIVATION_STR = "ReLU"

results = []

for data in DATASET_CONFIG:
    for model in MODEL_SET:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        print(f"\n=== {model} on {data} ===")
        start = time.time()
        train_loader, val_loader, test_loader = get_loaders(data=data,data_path=DATA_PATH,batch_size=BATCH_SIZE)
        model_class = getattr(models,model)
        model_name = model_class(in_channels=DATASET_CONFIG[data]["CHANNELS"],num_classes=DATASET_CONFIG[data]["NUM_CLASSES"],drop_rate=DATASET_CONFIG[data]["DROPOUT_RATE"],activation_str=ACTIVATION_STR).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_name.parameters(), lr=LEARNING_RATE)
        trainer = Trainer(model_name, criterion, optimizer, device)
        trainer.fit(train_loader, val_loader, epochs=EPOCHS)
        torch.save(model_name.state_dict(), f"{model}_{data}.pth")
        train_loss, train_acc, _, _, _ = trainer.evaluate(train_loader)
        if device.type == "cuda":
            train_memory = torch.cuda.max_memory_allocated() / (1024**2) 
        else :
            train_memory = 0
        duration = time.time() - start
        if device.type == "cuda":
            torch.cuda.synchronize()
        stop_watch = time.time()
        test_loss, test_acc, test_precision, test_recall, test_f1 = trainer.evaluate(test_loader)
        if device.type == "cuda":
            torch.cuda.synchronize()
        stop_watch = time.time() - stop_watch
        latency_per_sample = (stop_watch / len(test_loader.dataset)) * 1000
        results.append({
        "model_name": model,
        "dataset": data,
        "train_acc": train_acc, 
        "test_acc": test_acc,
        "precision": test_precision,
        "recall": test_recall,
        "f1": test_f1,
        "duration": duration,
        "memory_mb": train_memory,
        "latency_ms": latency_per_sample})


print("\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)
for r in results:
    print(f"{r['model_name']} on {r['dataset']} | "
          f"Train Acc: {r['train_acc']:.2f}% | " 
          f"Test Acc: {r['test_acc']:.2f}% | "
          f"Precision: {r['precision']:.4f} | "
          f"Recall: {r['recall']:.4f} | "
          f"F1: {r['f1']:.4f} | "
          f"Time: {r['duration']:.1f}s | "
          f"Memory: {r['memory_mb']:.1f}MB | "
          f"Latency: {r['latency_ms']:.3f}ms/sample")