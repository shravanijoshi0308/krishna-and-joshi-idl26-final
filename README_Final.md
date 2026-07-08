# MAI SS-2026 DL project - Final Assignment
## Operation Cyber-Histology: Total Pipeline Failure

### Team Members : 
- `Krishna Kapoor` - `10012546`
- `Shravani Joshi`-`10012627`

### Course
Introduction to Deep Learning (IDL) — THWS Würzburg-Schweinfurt  
Summer Semester 2026 | Prof. Dr. Magda Gregorová

### Assignment Overview 
A medical AI Company BioHealth Diagnotics lost their entier ML Pipeline due to sabotage.
Our task is to find, fix, correct and extend it. 

Starting from a deliberately sabotaged codebase where there are missing gradients, broken data splits, architecture bugs, and rigid hardcoded infrastructure, we fixed :
1. Identified and fixed distinct bugs across the codebase (Refer audit.md)
2. Built a proper config based training and evaluation pipeline. 
3. The three classical Architectures - AlexNet, VGG16 , ResNet18 are benchmarked across four datasets - cells, chest, lesions, orgs. 
4. For Green Initiative we designed and evaluated a custom lightweight architecture. 
5. For scarce Organs dataset we implemented and evaluated transfer learning. 

#### Repository Structure
```
krishna-and-joshi-idl26-final/
├── Code/
│   ├── data.py            # Loading dataset, splitting datset and normalization
│   ├── models.py          # Model architectures : AlexNet, VGG16, ResNet18, GreenNet 
│   ├── fit.py             # Trainer class (train_one_epoch, evaluate, fit)
│   ├── train.py           # Single run entry point, driven by config.json
│   ├── runner.py          # Automated runner 
│   ├── pretraining.py     # Part 3: transfer learning (scratch vs. transfer)
│   ├── config.json        # Configuration for single train.py runs
│   └── Data/              # Dataset 
├── AUDIT_LOG.md           # Audited bug list: Bug, root cause, fix, commit hash
├── README.md              # Readme file
├── REPORT.md              #  Results of Benchmarking, Green Initiative, Data-Scarcity analysis
└── .gitignore
```

##### Prequistes 
```
1. Python 3.11
2. PyTorch: cuda, mps or cpu - devices are auto detected
3. scikit-learn for precision , recall and F1 Metric. 
```

##### Installing Dependencies 
```bash
python3 -m venv .venv311
source .venv311/bin/activate   # macOS
venv311\Scripts\activate.      # Windows
pip install torch torchvision
pip install scikit-learn
```

##### Datasetup 
Download the datasets from the provided cloud link and place the .pt files inside Code/Data/:
```
Code/Data/
├── cells.pt
├── chest.pt
├── lesions.pt
├── orgs.pt
└── organs.pt
```

##### Models 
| Model | Description |
|---|---|
| AlexNet | Classic 2012 architecture (Krizhevsky et al.), adapted for 64×64 inputs |
| VGG16 | Configuration C (Simonyan & Zisserman) with Global Average Pooling |
| Resnet18 | Residual network (He et al., 2016) with skip connections | 
| GreenNet | Custom lightweight architecture (3 conv layers, single-layer classifier, Global Average Pooling) designed for the Green Initiative to minimize compute and memory while maintaining competitive accuracy

##### Running Pipeline

For single run on train, validate and test one model for one dataset combination: 
-> Edit `Code/config.json` and change model, dataset and hyperparameters:

```json
{
    "DATA": "cells",
    "DATA_PATH": "Data",
    "MODEL": "AlexNet",
    "CHANNELS": 3,
    "NUM_CLASSES": 8,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 0.0001,
    "EPOCHS": 10,
    "DROPOUT_RATE": 0.5,
    "ACTIVATION_STR": "ReLU"
}
```

Then run : 
```bash 
cd Code
python3 train.py 
```

This trains one model for one dataset and prints training and validation metrics and saves the trained weights as `{MODEL}_{DATA}.pth` and evaluates on the test set and records (accuracy, precision, recall and F1)

*The Valid Model Values are :* `AlexNet`, `VGG16` ,`ResNet18`, `GreenNet`
*The Valid Data Values are :* `cells(3ch/8cls)`, `chest(1ch/2cls)`, `lesions(3ch/7cls)`, `orgs (1ch/11cls)`.

##### Automated benchmark for all models x all datasets 
```bash
cd Code
python3 runner.py
```

This loops through all 4 models on all 4 datasets by training and evaluating each combination automatically. It generates a final summary tabke including test accuracy, precision, recall, peak memory consumption, training duration and inference latency per sample.This summary is used for both the main benchmark (part 1) and the Green Initiative efficiency comparison(part 2).

##### Transfer Learning 
```bash
cd Code
python3 pretraining.py
```

This runs the experiment in the scarce organs datasets(500 samples) :
1. Training Alexnet from scratch. 
2. Transfer learning - loading a model pretrained in the large orgs dataset, freezing the convolution backbone and fine tuning only the final classifer layer. 

Results are printed as benchmarking matrix comparing both approaches against the minimum accuracy target. 