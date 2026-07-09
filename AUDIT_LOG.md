# AUDIT_LOG.md — Operation Cyber-Histology Incident Audit

**Team:** Krishna Kapoor `10012546` & Shravani Joshi (10012627)
**Course:** MAI/IDL SS26, THWS Würzburg-Schweinfurt
**Repository:** krishna-and-joshi-idl26-final
**Branch:** final_changes_by_Krishna_&_Joshi

---

## Bug Summary Table

| # | File | Bug | Type | Error | Root Cause | Fix | Commit |
|---|---|---|---|---|---|---|---|
| 1 | data.py | Train and validation data leakage | Silent-bug | No crash, but val accuracy is not giving the accurate perfomance| `train_data` used full dataset without slicing | Added `[:val_start]` to `train_data` and `train_labels` | `c029cee` |
| 2 | data.py | Normalization was missing| Silent-bug | No crash, but training is slower and unstable because of raw pixel values | No mean/std are not computed anywhere in `get_loaders()` | Computed mean/std from train only and applied it to train/val/test | `5019843`, `ca0ccc0` |
| 3 | data.py | Filename pattern mismatch of dataset | Crash | `FileNotFoundError: No such file or directory: 'Data/cells_data.pt'` | Path built as `{data}_data.pt` instead of `{data}.pt` | Changed f-string to `f"{data}.pt"` | `032e666` |
| 4 | data.py | Mismatch of the lable [N,1]→[N] (train/val) | Crash | `RuntimeError: 0D or 1D target tensor expected, multi-target not supported` | Loaded Label as `[N,1]`, CrossEntropyLoss requires `[N]` | Added `torch.squeeze()` to train/val labels | `c9db021` |
| 5 | data.py | Mismatch of the lable [N,1]→[N] (test) | Crash | `RuntimeError: 0D or 1D target tensor expected, multi-target not supported` | `test_labels` also `[N,1]`, needed squeeze once test eval added | Added `torch.squeeze()` to test labels | `2fcc8a9` |
| 6 | fit.py | Missing `optimizer.zero_grad()` | Silent-bug | No crash, loss jumped up and down randomly instead gowing down smootly| Gradients accumulated across batches instead of resetting each step | Added `self.optimizer.zero_grad()` before each forward pass | `1b46f25` |
| 7 | fit.py | `sum` shadowing Python built-in | Silent-bug | No crash, risk of future breakage | Variable named `sum` silently replaced Python's built-in `sum()` | Renamed to `total` | `44d52a8` |
| 8 | models.py | Flatten size mismatch in AlexNet | Crash | `RuntimeError: linear(): input and weight.T shapes cannot be multiplied (32x3072 and 2048x1024)` | Hardcoded `2048` doesn't match 64×64 input (real size: 192×4×4 = 3072) | Changed `nn.Linear(2048, 1024)` to `nn.Linear(3072, 1024)` | `1b83c0d` |
| 9 | models.py | `in_channels`/`num_classes` ignored are in AlexNet| Silent-bug | No crash, model always built with 3 channels/11 classes regardless of config | Both hardcoded in `Conv2d`/`Linear`; `**kwargs` accepted but never read | Made `drop_rate`, `in_channels`, `num_classes` required parameters | `4797053` |
| 10 | models.py | Channel tracking in VGGBlock | Crash | `RuntimeError: Given groups=1, weight of size [64, 3, 3, 3], expected input[32, 64, 64, 64] to have 3 channels, but got 64 channels instead` | `current_in_channels`  is initialized once before loop but never updated inside it | Added `current_in_channels = out_channels` at end of each loop iteration | `af2824f` |
| 11 | models.py |  Wrong padding on 1×1 conv in VGGBlock | Silent-bug | No crash, feature maps silently grews from (64→66), effecting flatten size to 4608 | `padding=1` fixed for all kernels; rule `p=(k-1)/2` gives `p=0` for `k=1` | Added `padding_size = 0 if is_config_c_tail else 1` | `77b1add` |
| 12 | models.py | Missing `AdaptiveAvgPool2d` in VGG16  | Silent-bug | No crash, the size of input classifier every time upstream bug was fixed | In VGG16 GAP squashes each channel's grid number to one so the output is always exactly 512 numberss| Added `self.avgpool = nn.AdaptiveAvgPool2d((1,1))`, resized classifier to `512` | `55fe8ea` |
| 13 | models.py | Missing `return` in `forward()` of ResNet18 | Crash | `TypeError: cross_entropy_loss(): argument 'input' must be Tensor, not NoneType` | Last line `self.classifier(out)` had no `return`, function implicitly returned `None` | Added `return` before `self.classifier(out)` | `da19a98` |
| 14 | train.py | Hardcoded `drop_rate=0.99`  | Silent-bug | No crash — ~99% dropout caused severe underfitting | Literal `drop_rate=0.99` passed instead of `config["DROPOUT_RATE"]` | Changed to `drop_rate=config["DROPOUT_RATE"]` | `dc37157` |
| 15 | train.py | `activation_str=None` hardcoded | Silent-bug | No crash — `None` resolved to `nn.Identity`, removing all non-linearity | Literal `activation_str=None` passed instead of `config["ACTIVATION_STR"]` | Changed to `activation_str=config["ACTIVATION_STR"]` | `d165869` |
| 16 | train.py | Device check for MPS is missing( | Silent-bug | No crash — training silently ran on CPU on Apple Silicon, ignoring MPS GPU | Only checked `cuda`; no `mps` check for Apple Silicon | Added `mps` check before `cuda` in device detection | `6992dc0` |
| 17 | train.py | No model saving after training | Missing | No crash — trained weights lost after every run; transfer learning impossible | `torch.save()` never called anywhere in `main()` | Added `torch.save(model.state_dict(), filename)` after training | `d165869` |

---

## Detailed Bug Entries

### Bug 1 — Train and validation data leakage (data.py)
- **Error:** No crash. Validation accuracy looked good, but it was not correct the model had already seen those images during training, so it was being tested on data it had already memorized during training.
- **Root cause:** `train_data` was loading the entire dataset with no slicing. The validation set was correctly taking the last 10%, but training was using 100% — which included those same last 10%. Train data and validation data were overlapping.
- **Fix:** Added `[:val_start]` to `train_data` and `train_labels` so training only uses the first 90%, and validation gets the remaining 10% that training never touched.
- **Commit:** `c029cee

### Bug 2 — Normalization was missing| (data.py)
- **Error:** No crash. Slower, training was slow and unnormalized pixel values were directly fed into network.
- **Root cause:** No mean/std computation existed anywhere in `get_loaders()`.
- **Fix:** Computed `mean`/`std` from `train_data` only, applied `(x − mean) / std` to train, val, and test using those same train-derived statistics.
- **Commit:** `5019843` (train/val), `ca0ccc0` (test)

### Bug 3 — Filename pattern mismatch of dataset  (data.py)
- **Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'Data/cells_data.pt'`
- **Root cause:** The path built by `get_loaders()` was `f"{data}_data.pt"`, actual the files are named as `{data}.pt` no suffix of `_data`.
- **Fix:** Changed to `f"{data}.pt"`.
- **Commit:** `032e666`

### Bug 4 — Label shape mismatch train/val (data.py)
- **Error:** `RuntimeError: 0D or 1D target tensor expected, multi-target not supported`
- **Root cause:** The Labels were loaded as `[N,1]` each label is wrapped with an extra dimension. `CrossEntropyLoss` requires 1D targets `[N]` only.
- **Fix:** Added `torch.squeeze()` to `train_labels` and `val_labels`.
- **Commit:** `c9db021`

### Bug 5 — Mismatch of the lable [N,1]→[N] (test) (data.py)
- **Error:** `RuntimeError: 0D or 1D target tensor expected, multi-target not supported`
- **Root cause:** Same issue as Bug 4 but for `test_labels` — was noticed only after test evaluation was added
- **Fix:** Added `torch.squeeze()` to `test_labels`.
- **Commit:** `2fcc8a9`


### Bug 6 — Missing `optimizer.zero_grad()` (fit.py)
- **Error:** No crash, loss jumped up and down randomly instead gowing down smootly
- **Root cause:** `train_one_epoch()` never called `zero_grad()`, instead of resetting each step all the gradients from all prior batches are accumulated.
- **Fix:** Added `self.optimizer.zero_grad()` before each forward/backward pass.
- **Commit:** `1b46f25`

### Bug 7 — `sum` shadowing Python built-in (fit.py)
- **Error:** No crash. risk of future code calling `sum()` in the same scope would break unexpectedly.
- **Root cause:** Local variable named `sum` (`correct, sum = 0, 0`) silently replaced Python's built-in `sum()`.
- **Fix:** Renamed to `total`.
- **Commit:** `44d52a8`

### Bug 8 — Flatten size mismatch in AlexNet (models.py)
- **Error:** `RuntimeError: linear(): input and weight.T shapes cannot be multiplied (32x3072 and 2048x1024)`
- **Root cause:** AlexNet is original designed for 224×224 and this version uses 64×64, giving real flatten size 192 channels × 4×4 = 3072, not 2048.
- **Fix:** Changed `nn.Linear(2048, 1024)` to `nn.Linear(3072, 1024)`.
- **Commit:** `1b83c0d`

### Bug 9 — `in_channels`/`num_classes` ignored are in AlexNet (models.py)
- **Error:** No crash. Model always built with 3 channels/11 classes regardless of config — e.g. `NUM_CLASSES: 8` still produced 11-class output.
- **Root cause:** Both hardcoded in `Conv2d`/`Linear`; `**kwargs` accepted but never read.
- **Fix:** Made `drop_rate`, `in_channels`, `num_classes` required named parameters.
- **Commit:** `4797053`

### Bug 10 — Channel tracking in VGGBlock  (models.py)
- **Error:** `RuntimeError: Given groups=1, weight of size [64, 3, 3, 3], expected input[32, 64, 64, 64] to have 3 channels, but got 64 channels instead`
- **Root cause:** `current_in_channels` initialized once before the loop but never updated inside it, so every conv after the first reused the original input channel count.
- **Fix:** Added `current_in_channels = out_channels` at the end of each loop iteration.
- **Commit:** `af2824f`

### Bug 11 —  Wrong padding on 1×1 conv in VGGBlock (models.py)
- **Error:** No crash. Feature maps grew silently (64→66 pixels) instead of preserving size, cascading into incorrect flatten size (4608 instead of 2048).
- **Root cause:** `padding=1` fixed for all kernels. Rule `p=(k-1)/2` gives `p=0` for `k=1`, not `1`.
- **Fix:** Added `padding_size = 0 if is_config_c_tail else 1`.
- **Commit:** `77b1add`

### Bug 12 —  Missing `AdaptiveAvgPool2d` in VGG16 (models.py)
- **Error:** No crash. Classifier input size had to be manually recalculated every time an upstream bug was fixed (2048→4608→2048).
- **Root cause:** VGG16 flattened raw spatial output directly, coupling classifier size to exact input resolution.
- **Fix:** Added `self.avgpool = nn.AdaptiveAvgPool2d((1,1))`, resized first Linear layer to fixed `512`.
- **Commit:** `55fe8ea`

### Bug 13 — Missing `return` in `forward()` of ResNet18 (models.py)
- **Error:** `TypeError: cross_entropy_loss(): argument 'input' (position 1) must be Tensor, not NoneType`
- **Root cause:** Last line `self.classifier(out)` had no `return` keyword, function implicitly returned `None`.
- **Fix:** Added `return` before `self.classifier(out)`.
- **Commit:** `da19a98`

### Bug 14 — Hardcoded `drop_rate=0.99` (train.py)
- **Error:** No crash. ~99% dropout caused severe underfitting — near-random accuracy regardless of architecture.
- **Root cause:** Literal `drop_rate=0.99` passed to model instead of `config["DROPOUT_RATE"]`.
- **Fix:** Changed to `drop_rate=config["DROPOUT_RATE"]`.
- **Commit:** `dc37157`


### Bug 15 — Hardcoded `activation_str=None`  (train.py)
- **Error:** No crash. `None` resolved to `nn.Identity` — removing all non-linearity, collapsing the network to a single linear map.
- **Root cause:** Literal `activation_str=None` passed instead of `config["ACTIVATION_STR"]`.
- **Fix:** Changed to `activation_str=config["ACTIVATION_STR"]`.
- **Commit:** `d165869`

### Bug 16 — Device check for MPS is missing (train.py)
- **Error:** No crash. Training silently ran on CPU on Apple Silicon, ignoring the available MPS GPU.
- **Root cause:** `torch.device("cuda" if torch.cuda.is_available() else "cpu")` — no MPS check included.
- **Fix:** Added `mps` check: `torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")`.
- **Commit:** `6992dc0`

### Bug 17 — No model saving after training  (train.py)
- **Error:** No crash. Trained weights lost after every run — making transfer learning (Part 3) impossible without saved `.pth` files.
- **Root cause:** `torch.save()` never called anywhere in `main()`.
- **Fix:** Added `torch.save(model.state_dict(), f"{config['MODEL']}_{config['DATA']}.pth")` after `trainer.fit()`.
- **Commit:** `d165869`
---

## Known Limitations (identified, intentionally left as-is)

| Item | File | Reason left unfixed |
|---|---|---|
| `activation_str` not config-driven in AlexNet/VGG16 | models.py | Hardcoded `nn.ReLU(inplace=True)` inline; time constraints |
| `activation_str` reads module-level global in ResNet18 | models.py | Global `activation_str = "ReLU"` bypasses config; time constraints |
| Chest dataset train/test distribution shift | data.py | 74/26 train vs 62.5/37.5 test class balance; mitigated with dropout=0.6 |