"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

# Takes the name of the dataset, loads it from the disk and spilt it into training, validation and test datasets.
def get_loaders(data, data_path, batch_size, val_split=0.1):
    #d_path = Path(data_path) / f"{data}_data.pt"
    d_path = Path(data_path) / f"{data}.pt"
    data_dict = torch.load(d_path)

# Calculating and setting the data into training and validation     total_samples = data_dict['train_images'].shape[0]
    total_samples = data_dict['train_images'].shape[0]
    val_size = int(total_samples * val_split)
    val_start = total_samples - val_size
    
    #Adding torch.squeeze to the labels. 
    train_data = data_dict['train_images'][:val_start]
    #train_labels = data_dict['train_labels'][:val_start]
    train_labels = torch.squeeze(data_dict['train_labels'])[:val_start]
    val_data = data_dict['train_images'][val_start:]
    val_labels =  torch.squeeze(data_dict['train_labels'])[val_start:]
    
     # Normalization of training data before data feeding.
    mean = train_data.mean(dim=(0,2,3), keepdim=True)
    std = train_data.std(dim=(0,2,3), keepdim=True)
    train_data = (train_data - mean) / std
    val_data = (val_data - mean) / std
    test_data = (data_dict['test_images'] - mean) / std
    
    # The data is already split into training and test datasets
    # wrapping them into TensorDataset
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    #test_dataset = TensorDataset(data_dict['test_images'], data_dict['test_labels']) 
    # Squeezing test labels from [N,1] to [N]
    test_dataset = TensorDataset(test_data, torch.squeeze(data_dict['test_labels']))  
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader