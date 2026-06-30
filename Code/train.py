"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import json

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from fit import Trainer

def main():   
    with open("config.json", "r") as f:
        config = json.load(f)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    # To check on which device it is getting model is using. 
    print(f"Training executing on device: {device}")

    train_loader, val_loader, test_loader = get_loaders(data=config["DATA"], data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])

    model_class = getattr(models, config["MODEL"])
    model = model_class(in_channels=config["CHANNELS"], num_classes=config["NUM_CLASSES"], drop_rate=config["DROPOUT_RATE"], activation_str=config["ACTIVATION_STR"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    # To check which model is being trained 
    print(f"Training on model: {config['MODEL']}")

    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])
    # Adding model saving after training completes
    filename = f"{config['MODEL']}_{config['DATA']}.pth"
    torch.save(model.state_dict(), filename)
    # To check which model the is being used. 
    print("Model saved as :", filename)

    # Adding test set evaluation
    test_loss, test_acc, test_precision, test_recall, test_f1 = trainer.evaluate(test_loader)
    # Printing test results 
    print(f"\nTest Results | Loss: {test_loss:.4f} - Acc: {test_acc:.2f}% | "
      f"Precision: {test_precision:.4f} - Recall: {test_recall:.4f} - F1: {test_f1:.4f}")
    print("-" * 50)
    print("Testing Complete!")
    



if __name__ == "__main__":
    main()