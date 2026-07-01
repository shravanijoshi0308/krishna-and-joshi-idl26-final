"""This file runs the """

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
    "cells":   {"CHANNELS": 3, "NUM_CLASSES": 8},
    "chest":   {"CHANNELS": 1, "NUM_CLASSES": 2},
    "lesions": {"CHANNELS": 3, "NUM_CLASSES": 7},
    "orgs":    {"CHANNELS": 1, "NUM_CLASSES": 11},
}

MODEL_SET = ["AlexNet", "VGG16", "ResNet18"]
 
DATA_PATH = "Data"
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 10
DROPOUT_RATE = 0.5
ACTIVATION_STR = "ReLU"

results = []

for data in DATASET_CONFIG:
    for model in MODEL_SET:
        print(f"\n=== {model} on {data} ===")
        start = time.time()
        train_loader, val_loader, test_loader = get_loaders(data=data,data_path=DATA_PATH,batch_size=BATCH_SIZE)
        model_class = getattr(models,model)
        model_name = model_class(in_channels=DATASET_CONFIG[data]["CHANNELS"],num_classes=DATASET_CONFIG[data]["NUM_CLASSES"],drop_rate=DROPOUT_RATE,activation_str=ACTIVATION_STR).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_name.parameters(), lr=LEARNING_RATE)
        trainer = Trainer(model_name, criterion, optimizer, device)
        trainer.fit(train_loader, val_loader, epochs=EPOCHS)
        duration = time.time() - start
        test_loss, test_acc, test_precision, test_recall, test_f1 = trainer.evaluate(test_loader)
        results.append({
        "model_name": model,
        "dataset": data,
        "test_acc": test_acc,
        "precision": test_precision,
        "recall": test_recall,
        "f1": test_f1,
        "duration": duration})

print("\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)
for r in results:
    print(f"{r['model_name']} on {r['dataset']} | "
          f"Acc: {r['test_acc']:.2f}% | "
          f"P: {r['precision']:.4f} | "
          f"R: {r['recall']:.4f} | "
          f"F1: {r['f1']:.4f} | "
          f"Time: {r['duration']:.1f}s")