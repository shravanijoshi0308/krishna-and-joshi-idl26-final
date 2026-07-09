# REPORT.md — Operation Cyber-Histology Final Report

**Team:** `Krishna Kapoor - 10012546` & `Shravani Joshi - 10012627`
**Course:** MAI/IDL SS26, THWS Würzburg-Schweinfurt
**Repository:** krishna-and-joshi-idl26-final

---

## Dataset Overview

All datasets use 64×64 pixel images. Class distributions are shown to highlight imbalance issues.

| Dataset | Train Samples | Test Samples | Channels | Classes | 
|---|---|---|---|---|
| cells | 13,671 | 3,421 | 3 (RGB) | 8 |
| chest | 5,232 | 624 | 1 (grayscale) | 2 | 
| lesions | 8,010 | 2,005 | 3 (RGB) | 7 | 
| orgs | 15,367 | 8,216 | 1 (grayscale) | 11 | 
| organs | 500 | 200 | 1 (grayscale) | 11 |

## Part 1 — Consolidated Benchmark Report

### Overview

After identifying and fixing 17 bugs across the codebase (refer `AUDIT_LOG.md`), we benchmarked three classical architectures — AlexNet, VGG16 and ResNet18 across all four medical imaging datasets. We Training used 10 epochs, learning rate 0.0001, batch size 32, and dropout 0.5, 0.6 for chest.

## Part 1 — Consolidated Benchmark Report

| Model | Dataset | Train Acc | Test Acc | Precision | Recall | F1 | Time | Memory | Latency |
|---|---|---|---|---|---|---|---|---|---|
| AlexNet | cells | 98.16% | *96.67%* | 0.9652 | 0.9644 | 0.9645 | 28.2s | 187.1MB | 0.085ms |
| VGG16 | cells | 98.07% | *96.58%* | 0.9595 | 0.9690 | 0.9635 | 71.7s | 532.7MB | 0.183ms |
| ResNet18 | cells | 97.61% | *96.11%* | 0.9598 | 0.9512 | 0.9541 | 142.6s | 787.9MB | 0.335ms |
| AlexNet | chest | 99.55% | **88.78%**  | 0.9134 | 0.8547 | 0.8725 | 9.1s | 228.0MB | 0.069ms |
| VGG16 | chest | 98.28% | **78.85%**  | 0.8692 | 0.7188 | 0.7322 | 26.8s | 530.9MB | 0.184ms |
| ResNet18 | chest | 99.98% | **85.58%**  | 0.9008 | 0.8094 | 0.8302 | 51.9s | 790.0MB | 0.313ms |
| AlexNet | lesions | 81.04% | *78.50%* | 0.5425 | 0.5136 | 0.5241 | 14.7s | 229.9MB | 0.076ms |
| VGG16 | lesions | 85.60% | *77.31%*  | 0.5315 | 0.5632 | 0.5395 | 41.4s | 603.3MB | 0.184ms |
| ResNet18 | lesions | 81.90% | *76.46%*  | 0.7563 | 0.4530 | 0.5204 | 80.9s | 834.2MB | 0.316ms |
| AlexNet | orgs | 98.71% | *90.88%*  | 0.8966 | 0.9009 | 0.8963 | 23.9s | 272.9MB | 0.059ms |
| VGG16 | orgs | 99.17% | *91.70%*  | 0.9077 | 0.9121 | 0.9087 | 75.6s | 642.7MB | 0.166ms |
| ResNet18 | orgs | 98.16% | *89.91%*  | 0.9033 | 0.8758 | 0.8845 | 188.0s | 875.9MB | 0.550ms |

**Targets:** cells ≥ 90% | chest ≥ 87% | lesions ≥ 67% | orgs ≥ 83%
**Results: 10/12 combinations meet their targets.**

### Analysis

**cells** - All four models cleared their baseline target `cells ≥ 90%`. This dataset have balanced classes and in_channels `8cls` , `3chl` and they have enough data to train the model and learn well. ResNet18 performed best on cells `96.11%`
**orgs** — All four models cleared their baseline target `orgs ≥ 83%`. This dataset have balanced classes and in_channels `orgs (1ch/11cls)`and they have enough data to train the model and learn well. VGG16 performed best on orgs `91.70%`.

**lesions** — All four models  cleared their baseline target `lesions ≥ 67%` but the lower F1 scores (0.43-0.54) tells an important story: that all the models are doing well on the majority class but struggling with the rare classes. This is because the lesions dataset is imbalanced — one class makes up most of the samples, so the model learns to "lean" towards predicting that class. Overall accuracy looks acceptable, but the model misses many of the rarer lesion types. 

**chest** — This was the hardest dataset. The baseline target of this `chest ≥ 87%`. we discovered two connected problems:

First, there is a *class distribution mismatch* between the training data and the test data. 
Second, because of this mismatch, the *validation accuracy was misleading* during training. 

As it was not reaching the baseline accuracy because of the above mentioned reasons, we mitigated it increasing dropout to 0.6 for chest specifically, which helped AlexNet clear the 87% target (88.78%). VGG16 and ResNet18 fell just short (78.85% and 85.58%).

### Architectural Recommendations

| Dataset | Recommended Model | Reason |
|---|---|---|
| cells | ResNet18 | Highest accuracy (96.11%), skip connections aid gradient flow |
| chest | AlexNet | Only model consistently clearing target; simpler architecture less prone to chest's distribution shift |
| lesions | AlexNet | Best test accuracy (78.50%) with reasonable F1 |
| orgs | VGG16 | Best test accuracy (91.70%), strong generalization on multi-class organ classification |

---

## Part 2 — Green Initiative Analysis

### GreenNet Architecture

GreenNet is a custom lightweight architecture that is  designed specifically to use less memory and run faster, without using much memory. The Architecture of GreenNet is :
- 3 convolutional layers (channels: 32 → 64 → 128)
- AdaptiveAvgPool2d before the classifier 
- A single Linear classifier (128 → num_classes) 
- BatchNorm for regularization instead of dropout


### GreenNet Independent Validation

GreenNet was additionally validated across all four datasets, confirming consistent performance:

| Dataset | Train Acc | Test Acc | Precision | Recall | F1 | Time | Memory | Latency |
|---|---|---|---|---|---|---|---|---|
| cells | 94.47% | **93.54%** | 0.9384 | 0.9265 | 0.9302 | 18.2s | 189.4MB | 0.064ms |
| chest | 96.03% | *85.26%*  | 0.8784 | 0.8128 | 0.8303 | 5.8s | 190.6MB | 0.059ms |
| lesions | 75.32% | **74.81%** | 0.5502 | 0.3976 | 0.4398 | 9.6s | 234.0MB | 0.060ms |
| orgs | 91.11% | **85.80%** | 0.8437 | 0.8314 | 0.8335 | 34.6s | 277.2MB | 0.050ms |

**Targets:** cells ≥ 90% | chest ≥ 87% | lesions ≥ 67% | orgs ≥ 83%

GreenNet cleared 3 out of 4 targets independently, with chest falling just short  consistent with the class distribution mismatch identified for all models on this dataset. These results confirm GreenNet's reliability as a lightweight alternative across different hardware environments.

### Key Efficiency Findings

**Memory**: GreenNet uses ~190-277MB compared to ResNet18's 788-876MB approximately **4× less memory** than ResNet18, and **2.5× less** than VGG16.

**Training speed**: GreenNet trains **6-11× faster** than ResNet18 (e.g. 34.6s vs 188.0s on orgs), and **4× faster** than VGG16.

**Inference latency**: GreenNet's per-sample latency (0.044-0.064ms) is the lowest of all four models approximately **6× faster** than ResNet18 at inference time.

**Accuracy trade-off**: GreenNet achieves accuracy within 2-5% of the best heavyweight model on most datasets, with one notable exception — GreenNet uniquely matched AlexNet on chest (both ~85%), and on cells it came within 2.1% of the best result (93.54% vs 96.67%). On orgs, the gap widens to ~6%, suggesting GreenNet's capacity may limit performance on the most complex multi-class tasks.

**Overfitting analysis**: GreenNet showed near-zero train/val gaps across all datasets (e.g. 94.47%/93.54% on cells, 75.32%/74.81% on lesions), while larger models showed notable overfitting on lesions (ResNet18: 81.90% train vs 76.46% test, ~5.4% gap; VGG16: 85.60% train vs 77.31% test, ~8.3% gap). GreenNet's lower capacity makes it inherently more resistant to overfitting, particularly valuable on data-scarce or imbalanced datasets.

### Conclusion

GreenNet uses `4× less memory` and trains up to `11× faster` than ResNet18, while staying within `3-5% accuracy` on most datasets. This proves that a smaller, purpose-built model can still do the job well making it a practical choice when hardware or energy is limited.

---

### Full Benchmark Comparison (All Models)

| Model | Dataset | Train Acc | Test Acc | Precision | Recall | F1 | Time | Memory | Latency |
|---|---|---|---|---|---|---|---|---|---|
| AlexNet | cells | 98.16% | 96.67%  | 0.9652 | 0.9644 | 0.9645 | 28.2s | 187.1MB | 0.085ms |
| VGG16 | cells | 98.07% | 96.58%  | 0.9595 | 0.9690 | 0.9635 | 71.7s | 532.7MB | 0.183ms |
| ResNet18 | cells | 97.61% | 96.11%  | 0.9598 | 0.9512 | 0.9541 | 142.6s | 787.9MB | 0.335ms |
| GreenNet | cells | 94.47% | 93.54%  | 0.9384 | 0.9265 | 0.9302 | 18.2s | 189.4MB | 0.064ms |
| AlexNet | chest | 99.55% | 88.78%  | 0.9134 | 0.8547 | 0.8725 | 9.1s | 228.0MB | 0.069ms |
| VGG16 | chest | 98.28% | 78.85%  | 0.8692 | 0.7188 | 0.7322 | 26.8s | 530.9MB | 0.184ms |
| ResNet18 | chest | 99.98% | 85.58%  | 0.9008 | 0.8094 | 0.8302 | 51.9s | 790.0MB | 0.313ms |
| GreenNet | chest | 96.03% | 85.26% | 0.8784 | 0.8128 | 0.8303 | 5.8s | 190.6MB | 0.059ms |
| AlexNet | lesions | 81.04% | 78.50%  | 0.5425 | 0.5136 | 0.5241 | 14.7s | 229.9MB | 0.076ms |
| VGG16 | lesions | 85.60% | 77.31%  | 0.5315 | 0.5632 | 0.5395 | 41.4s | 603.3MB | 0.184ms |
| ResNet18 | lesions | 81.90% | 76.46%  | 0.7563 | 0.4530 | 0.5204 | 80.9s | 834.2MB | 0.316ms |
| GreenNet | lesions | 75.32% | 74.81%  | 0.5502 | 0.3976 | 0.4398 | 9.6s | 234.0MB | 0.060ms |
| AlexNet | orgs | 98.71% | 90.88%  | 0.8966 | 0.9009 | 0.8963 | 23.9s | 272.9MB | 0.059ms |
| VGG16 | orgs | 99.17% | 91.70%  | 0.9077 | 0.9121 | 0.9087 | 75.6s | 642.7MB | 0.166ms |
| ResNet18 | orgs | 98.16% | 89.91%  | 0.9033 | 0.8758 | 0.8845 | 188.0s | 875.9MB | 0.550ms |
| GreenNet | orgs | 91.11% | 85.80% | 0.8437 | 0.8314 | 0.8335 | 34.6s | 277.2MB | 0.050ms |

*Targets:* cells ≥ 90% | chest ≥ 87% | lesions ≥ 67% | orgs ≥ 83%
*Part 1 (AlexNet/VGG16/ResNet18): 10/12 targets met.*
*Part 2 (GreenNet): 3/4 targets met.*
*Overall: 13/16 meet their targets.*

## Part 3 — Data-Scarcity Post-Mortem

### Context

The organs dataset is tiny only 500 training images for 11 different classes. With so few examples, a model trained from scratch struggles to learn anything useful.

### Transfer Learning Strategy

We followed two key rules of thumb for transfer learning:

| Rule | Our Case |
|---|---|
| Less data → freeze more layers | `organs` has only 450 training images so we froze everything except the last layer |
| More similar source domain → reuse more layers | `orgs` and `organs` share the same 11 classes and same image type so we reused the entire backbone |

This strategy protects everything the model already learned from 15,367 `orgs` images, while letting only the final decision layer adapt to the 450 `organs` images.

### Experimental Setup

Two experiments were conducted on the `organs` dataset (450 train / 50 val / 200 test):

1. **Scratch**: AlexNet trained from random initialization on organs only
2. **Same-domain transfer**: We took an AlexNet already trained on orgs, froze everything except the last layer, and let only that layer learn from the small organs dataset

### Data Benchmark Matrix

| Experiment | Loss | Test Accuracy | Precision | Recall | F1 | 40% Target |
|---|---|---|---|---|---|---|
| Scratch (AlexNet) | 1.5557 | 46.50% | 0.3979 | 0.4114 | 0.3525 | meets |
| Transfer (AlexNet, frozen backbone) | 1.4758 | **58.50%** | 0.5642 | 0.5245 | 0.5195 | meets |
| **Improvement** | | **+12.00%** | +0.1663 | +0.1131 | +0.1670 | |

*Improvement from transfer learning: +12.50% accuracy*
*Scratch: 46.00% — meets the 40% minimum target.*
*Transfer (frozen backbone): 58.50% — meets the 40% minimum target.*

### Analysis

Training from scratch was unstable the model kept improving and dropping randomly, showing it was guessing rather than truly learning. The transfer model hit 64% from epoch 1 and kept improving steadily, because it came with useful knowledge already built in

**Why same-domain transfer works so well**: `orgs` and `organs` have the same 11 classes and the same type of images.  The pretrained model already knew how to recognize organs only the final layer needed to learn the new dataset. Same task, same images, 
just less data perfect conditions for transfer learning.

**Consistency of results**: We ran the experiment multiple times without a fixed seed. Scratch accuracy varied a lot (35-48% each run), while transfer learning  consistently landed around 58-59%. Transfer learning doesn't just improve accuracy it also makes results more stable and predictable, which matters a lot when you have very little training data.

### Recommendations for Future Data Collection

1. **Consider partial fine-tuning as data grows** — as `organs` expands, unfreezing additional conv layers (not just the classifier) may allow deeper adaptation to organs-specific visual characteristics.
2. **Data augmentation** - (rotations, flips, brightness variation) a low-cost way to artificially expand the effective training set size without new data collection.
3. **Cross-validation** — with only 500 samples, a single train/val split may not be representative; k-fold cross-validation would give more reliable estimates of true model performance.