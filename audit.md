# AUDIT_LOG.md — Operation Cyber-Histology Incident Audit

**Team:** Krishna Kapoor , Shravani Joshi]
**Course:** MAI/IDL SS26, THWS
**Repository:** [repo link]

> Draft for review — fill in actual commit hashes from `git log --oneline -- <filename>` before final submission. Lecture references are to M. Gregorová's IDL SS26 notes (March 2026); a few are marked **[verify]** where the exact section wasn't directly quoted during debugging and should be checked against your own copy before citing it in the viva.

---

## data.py

### Bug 1 — Dataset filename pattern mismatch
- **Manifests as:** `FileNotFoundError: [Errno 2] No such file or directory: 'Data/cells_data.pt'`
- **Root cause:** `get_loaders()` built the file path as `f"{data}_data.pt"`, but the actual dataset files are named `{data}.pt` (e.g. `cells.pt`), with no `_data` suffix.
- **Fix:** Changed the f-string to `f"{data}.pt"`.
- **Lecture connection:** Not a theory bug — file I/O / naming convention mismatch.

### Bug 2 — Train/validation data leakage
- **Manifests as:** No crash. Validation accuracy was inflated/unreliable because the model had already seen validation samples during training.
- **Root cause:** `train_data`/`train_labels` used the full dataset (`data_dict['train_images']`) with no slicing, while `val_data`/`val_labels` were correctly sliced with `[val_start:]`. This meant every validation sample was also present in the training set.
- **Fix:** Added `[:val_start]` slicing to `train_data` and `train_labels` so train and validation sets are disjoint.
- **Lecture connection:** §6.1.2 + §5.4.1 — *"μⱼ and σⱼ must be computed on the training set only... computing statistics on the full dataset would leak information... a subtle but consequential data leakage error."* Same principle applies to the split itself, not just normalization stats.

### Bug 3 — Missing normalization
- **Manifests as:** No crash. Training was slower/less stable than expected; raw pixel values fed directly into the network.
- **Root cause:** No mean/std computation or normalization step existed anywhere in `get_loaders()`.
- **Fix:** Computed `mean`/`std` from `train_data` only (`dim=(0,2,3), keepdim=True`), then applied `(x - mean) / std` to train, val, **and** test using those same train-derived values (avoiding leakage from val/test statistics).
- **Lecture connection:** §5.4.1 — explains unnormalized features produce elongated, ill-conditioned loss contours, causing oscillation/slow convergence; the z-score fix (eq. 5.8) is exactly what was implemented.

### Bug 4 — Label shape mismatch ([N,1] instead of [N])
- **Manifests as:** `RuntimeError: 0D or 1D target tensor expected, multi-target not supported`, crashing inside `CrossEntropyLoss` during training and validation.
- **Root cause:** `train_labels`/`val_labels` were loaded directly with shape `[N, 1]` (each label wrapped in an extra dimension). `CrossEntropyLoss` requires 1D targets (`[N]`).
- **Fix:** Wrapped label extraction in `torch.squeeze(...)` for both `train_labels` and `val_labels`.
- **Lecture connection:** Tensor shape contract for the loss function — see §8 (Classification & MLE / cross-entropy) for the expected target format. **[verify exact equation]**

---

## fit.py

### Bug 5 — Missing `optimizer.zero_grad()`
- **Manifests as:** No crash. Training was numerically unstable — loss behaved erratically rather than improving cleanly.
- **Root cause:** `train_one_epoch()` never called `zero_grad()` before the forward/backward pass. PyTorch accumulates gradients by default, so each batch's gradients added onto leftover gradients from every prior batch instead of starting fresh.
- **Fix:** Added `self.optimizer.zero_grad()` before computing `outputs`/`loss` each batch.
- **Lecture connection:** §4.6 "Technical Implementation" — *"The backward pass initialises all `*.grad` to zero... and accumulates gradients from right to left."* Accumulation is correct **within** one backward pass; the bug was not resetting that accumulator between batches.

### Bug 6 — `sum` shadowing Python's built-in
- **Manifests as:** No crash in this specific function. Latent risk: any future code in the same scope calling the real `sum()` builtin would break.
- **Root cause:** A local variable was named `sum` (`correct, sum = 0, 0`), which silently replaces Python's built-in `sum()` function for the rest of that scope.
- **Fix:** Renamed the variable to `total`.
- **Lecture connection:** Pure Python hygiene — not covered in lecture notes, general coding practice (variable shadowing).

---

## models.py — AlexNet

### Bug 7 — Classifier flatten size mismatch
- **Manifests as:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x3072 and 2048x1024)`
- **Root cause:** The original AlexNet (Krizhevsky et al., 2012) was designed for 224×224 inputs; this version's conv/pooling stack is adapted for 64×64 inputs, producing a real flattened size of 192 channels × 4×4 spatial = 3072 — not the hardcoded 2048.
- **Fix:** Changed `nn.Linear(2048, 1024)` to `nn.Linear(3072, 1024)`.
- **Lecture connection:** §8.2 — convolution/pooling output-size formula `Hout = ⌊(H + 2p − k)/s⌋ + 1`, applied across the full conv stack to derive the real flattened dimension. **[verify exact section]**

### Bug 8 — `in_channels`/`num_classes` ignored
- **Manifests as:** No crash. Model always built with 3 channels / 11 classes regardless of `config.json` — e.g. setting `NUM_CLASSES: 8` still produced an 11-class output layer.
- **Root cause:** `__init__` accepted `**kwargs` but never read `in_channels`/`num_classes` from it — both were hardcoded directly into `Conv2d`/`Linear`.
- **Fix:** Made `drop_rate`, `in_channels`, `num_classes` required named parameters (instead of optional kwargs with silent fallbacks), so a missing value now fails loudly with `TypeError` instead of silently defaulting.
- **Lecture connection:** §8.2.5/8.2.6 — a filter needs one slice per input channel and one output unit per class; hardcoding breaks any dataset with a different channel count or class count (e.g. grayscale `chest`/`orgs` datasets, `C=1`).

---

## models.py — VGG16 / VGGBlock

### Bug 9 — `VGGBlock` reused original `in_channels` instead of updating after each conv
- **Manifests as:** `RuntimeError: Given groups=1, weight of size [64, 3, 3, 3], expected input[32, 64, 64, 64] to have 3 channels, but got 64 channels instead`
- **Root cause:** `current_in_channels` was initialized once before the loop but never reassigned inside it, so every conv after the first in a block was built expecting the block's original input channel count instead of the previous conv's actual output channel count.
- **Fix:** Added `current_in_channels = out_channels` at the end of each loop iteration.
- **Lecture connection:** General conv channel-matching principle — each conv layer's `in_channels` must equal the previous layer's `out_channels`. **[verify exact section in your notes]**

### Bug 10 — Wrong padding on the 1×1 "tail" convolution
- **Manifests as:** No crash, but silently wrong spatial sizes — feature maps grew by 2 at each affected block instead of staying constant, which cascaded into an incorrect flatten size (`4608` instead of the architecturally correct `2048`).
- **Root cause:** `padding` was fixed at `1` regardless of `kernel_size`. The "same padding" rule is `p = (k−1)/2`: for `k=3`, `p=1` (correct); for `k=1` (VGG "Configuration C"'s tail conv), `p` should be `0`, not `1`.
- **Fix:** Added `padding_size = 0 if is_config_c_tail else 1`, computed per conv inside the loop, same as `kernel_size`.
- **Lecture connection:** §8.2.3 — *"Valid padding (p=0) applies no padding at all... Same padding sets p = (k−1)/2, which adds enough zeros to keep the output the same size as the input for odd filter sizes."*

### Bug 11 — Missing `AdaptiveAvgPool2d`
- **Manifests as:** No crash, but the classifier's required input size depended on exact input resolution and had to be manually recalculated every time an upstream bug was fixed (`2048` → `4608` → `2048` again).
- **Root cause:** Unlike `ResNet18` (which includes `nn.AdaptiveAvgPool2d((1,1))`), `VGG16` flattened the raw spatial output of `self.features` directly, making the classifier fragile to any change in input resolution or upstream architecture.
- **Fix:** Added `self.avgpool = nn.AdaptiveAvgPool2d((1,1))`, called between `self.features(x)` and `torch.flatten` in `forward()`; resized the classifier's first `Linear` layer from `2048`/`4608` to a fixed `512` (the channel count).
- **Lecture connection:** §8.5.3 Global Average Pooling — *"GAP... is spatial-dimension-agnostic — the same GAP layer works for any input resolution."*

---

## models.py — ResNet18 / ResBlock

### Bug 12 — `forward()` missing `return`
- **Manifests as:** `TypeError: cross_entropy_loss(): argument 'input' (position 1) must be Tensor, not NoneType`
- **Root cause:** The last line of `forward()` was `self.classifier(out)` with no `return`, so the function implicitly returned `None` instead of the classifier's output.
- **Fix:** Added `return` before `self.classifier(out)`.
- **Lecture connection:** §3.2, eq. 3.5–3.7 — the network is defined as a composition of functions ending in an explicit output. A forward pass that doesn't return anything silently breaks that composition.

---

## train.py

### Bug 13 — `drop_rate`/`activation_str` hardcoded, ignoring config
- **Manifests as:** No crash. Models always trained with `drop_rate=0.99` (near-total dropout) and `activation_str=None` (→ `nn.Identity`, no non-linearity), regardless of `config.json`.
- **Root cause:** The model-building line in `main()` passed literal hardcoded values instead of reading from the loaded `config` dict.
- **Fix:** Changed to `drop_rate=config["DROPOUT_RATE"]`, `activation_str=config["ACTIVATION_STR"]`.
- **Lecture connection:** §6.3.2 — *"Typical values of p range from 0.1 to 0.5."* At 0.99, ~99% of neurons are zeroed every forward pass — severe underfitting, not regularization. For the Identity-activation half: §2.3/§2.6 — without non-linearity, stacking any number of linear layers is mathematically equivalent to one linear map (Universal Approximation requires a nonlinear activation).

### Bug 14 — Device check only detects `cuda`, ignores Apple Silicon `mps`
- **Manifests as:** No crash. Training silently ran on CPU on Apple Silicon hardware that has a usable GPU, with no indication anything was suboptimal.
- **Root cause:** `torch.device("cuda" if torch.cuda.is_available() else "cpu")` only checks for Nvidia GPUs.
- **Fix:** Changed to `torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")`.
- **Lecture connection:** Not theory-related — hardware/infrastructure optimization.

---

## Known limitations (identified, intentionally left as-is)

These were found during the audit but left unfixed due to time constraints rather than overlooked. Documenting them honestly here rather than silently omitting them.

- **AlexNet & VGG16 — `activation_str` not config-driven.** Both still hardcode `nn.ReLU(inplace=True)` directly inline in `self.features`/`self.classifier`, rather than building the activation from `kwargs.get("activation_str", ...)` the way `drop_rate`/`in_channels`/`num_classes` now do. Changing `ACTIVATION_STR` in `config.json` has no effect on these two models.
- **ResNet18 — `activation_str` read from a module-level global, not config.** A different mechanism from the above (a global variable `activation_str = "ReLU"` defined at the top of `models.py`, rather than an inline hardcode), but the same end result: `config.json`'s `ACTIVATION_STR` is silently ignored.
- **Lecture connection (both):** §2.3/§2.6 — same activation-function theory as Bug 13's Identity issue; these are scope decisions, not a misunderstanding of why activation choice matters.

## Pending — not yet addressed

- **`test_labels` squeeze** (`data.py`) — same `[N,1]`→`[N]` fix as Bug 4, deferred until the test-evaluation step is built, since `test_labels` isn't used anywhere yet.
- **Model saving** (`train.py`) — no `torch.save(...)` call exists anywhere; trained weights are not persisted after a run.
- **Test-set evaluation** (`train.py`) — `test_loader` is built by `get_loaders()` but immediately discarded (`_`); no code currently computes accuracy/precision/recall/F1 on the test set, which is what the assignment's accuracy targets are actually measured against.


FINAL RESULTS SUMMARY
==================================================
AlexNet on cells | Acc: 95.41% | P: 0.9541 | R: 0.9464 | F1: 0.9486 | Time: 26.0s
VGG16 on cells | Acc: 96.55% | P: 0.9600 | R: 0.9642 | F1: 0.9614 | Time: 69.8s
ResNet18 on cells | Acc: 96.49% | P: 0.9569 | R: 0.9620 | F1: 0.9584 | Time: 135.2s
AlexNet on chest | Acc: 82.85% | P: 0.8828 | R: 0.7739 | F1: 0.7935 | Time: 8.5s
VGG16 on chest | Acc: 89.26% | P: 0.9267 | R: 0.8568 | F1: 0.8769 | Time: 25.7s
ResNet18 on chest | Acc: 87.50% | P: 0.8993 | R: 0.8402 | F1: 0.8576 | Time: 50.8s
AlexNet on lesions | Acc: 76.86% | P: 0.5044 | R: 0.5021 | F1: 0.4766 | Time: 15.5s
VGG16 on lesions | Acc: 78.05% | P: 0.5207 | R: 0.5188 | F1: 0.4922 | Time: 40.8s
ResNet18 on lesions | Acc: 73.77% | P: 0.5469 | R: 0.5447 | F1: 0.4672 | Time: 79.4s
AlexNet on orgs | Acc: 89.72% | P: 0.8876 | R: 0.8886 | F1: 0.8873 | Time: 23.6s
VGG16 on orgs | Acc: 92.09% | P: 0.9090 | R: 0.9131 | F1: 0.9096 | Time: 74.0s
ResNet18 on orgs | Acc: 92.22% | P: 0.9147 | R: 0.9136 | F1: 0.9130 | Time: 152.9s


on mps 

==================================================
FINAL RESULTS SUMMARY
==================================================
AlexNet on cells | Train Acc: 94.31% | Test Acc: 93.54% | Precision: 0.9315 | Recall: 0.9222 | F1: 0.9250 | Time: 32.1s | Memory: 0.0MB | Latency: 0.455ms/sample
VGG16 on cells | Train Acc: 98.07% | Test Acc: 97.46% | Precision: 0.9783 | Recall: 0.9725 | F1: 0.9748 | Time: 166.7s | Memory: 0.0MB | Latency: 1.178ms/sample
ResNet18 on cells | Train Acc: 90.80% | Test Acc: 91.00% | Precision: 0.9454 | Recall: 0.9034 | F1: 0.9163 | Time: 449.9s | Memory: 0.0MB | Latency: 2.837ms/sample
GreenNet on cells | Train Acc: 88.21% | Test Acc: 88.19% | Precision: 0.8750 | Recall: 0.8630 | F1: 0.8677 | Time: 22.8s | Memory: 0.0MB | Latency: 0.080ms/sample
AlexNet on chest | Train Acc: 91.21% | Test Acc: 71.15% | Precision: 0.8421 | Recall: 0.6154 | F1: 0.5938 | Time: 14.6s | Memory: 0.0MB | Latency: 0.209ms/sample
VGG16 on chest | Train Acc: 97.69% | Test Acc: 88.62% | Precision: 0.9035 | Recall: 0.8568 | F1: 0.8721 | Time: 72.4s | Memory: 0.0MB | Latency: 1.857ms/sample
ResNet18 on chest | Train Acc: 99.53% | Test Acc: 81.89% | Precision: 0.8744 | Recall: 0.7620 | F1: 0.7805 | Time: 178.1s | Memory: 0.0MB | Latency: 2.761ms/sample
GreenNet on chest | Train Acc: 89.11% | Test Acc: 71.96% | Precision: 0.8166 | Recall: 0.6295 | F1: 0.6169 | Time: 7.3s | Memory: 0.0MB | Latency: 0.187ms/sample
AlexNet on lesions | Train Acc: 72.80% | Test Acc: 71.12% | Precision: 0.3588 | Recall: 0.3443 | F1: 0.3110 | Time: 23.9s | Memory: 0.0MB | Latency: 0.243ms/sample
VGG16 on lesions | Train Acc: 69.41% | Test Acc: 68.08% | Precision: 0.3169 | Recall: 0.3388 | F1: 0.3064 | Time: 264.0s | Memory: 0.0MB | Latency: 0.872ms/sample
ResNet18 on lesions | Train Acc: 73.96% | Test Acc: 71.82% | Precision: 0.4882 | Recall: 0.4290 | F1: 0.4336 | Time: 236.5s | Memory: 0.0MB | Latency: 2.709ms/sample
GreenNet on lesions | Train Acc: 70.07% | Test Acc: 70.17% | Precision: 0.3646 | Recall: 0.2729 | F1: 0.2753 | Time: 12.0s | Memory: 0.0MB | Latency: 0.247ms/sample
AlexNet on orgs | Train Acc: 94.00% | Test Acc: 88.05% | Precision: 0.8658 | Recall: 0.8672 | F1: 0.8642 | Time: 2983.7s | Memory: 0.0MB | Latency: 0.431ms/sample
VGG16 on orgs | Train Acc: 96.49% | Test Acc: 90.13% | Precision: 0.8903 | Recall: 0.8959 | F1: 0.8894 | Time: 1987.8s | Memory: 0.0MB | Latency: 0.859ms/sample
ResNet18 on orgs | Train Acc: 97.27% | Test Acc: 90.03% | Precision: 0.8922 | Recall: 0.8873 | F1: 0.8858 | Time: 405.5s | Memory: 0.0MB | Latency: 2.654ms/sample
GreenNet on orgs | Train Acc: 83.60% | Test Acc: 78.40% | Precision: 0.7607 | Recall: 0.7423 | F1: 0.7487 | Time: 17.4s | Memory: 0.0MB | Latency: 0.191ms/sample
(.venv311) shravanijoshi@Shravanis-MacBook-Air Code % python3 pretraining.py 