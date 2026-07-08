# Testing to check how many values does our datasets have 
import torch
d = torch.load("Data/cells.pt", weights_only=False)
print(torch.unique(d["train_labels"]))


# config file missing
# fix : create a config file 
(.venv311) shravanijoshi@Mac Code % python3 train.py
/Users/shravanijoshi/Desktop/krishna-and-joshi-idl26-final/.venv311/lib/python3.11/site-packages/torch/_subclasses/functional_tensor.py:362: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at /Users/runner/work/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)
  cpu = _conversion_method_template(device=torch.device("cpu"))
Traceback (most recent call last):
  File "/Users/shravanijoshi/Desktop/krishna-and-joshi-idl26-final/Code/train.py", line 33, in <module>
    main()
  File "/Users/shravanijoshi/Desktop/krishna-and-joshi-idl26-final/Code/train.py", line 16, in main
    with open("config.json", "r") as f:
         ^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'config.json'

# ErrNo  FileNotFoundError:  No such file or directory: 'Data/cells_data.pt'
# Fix :
The code constructed the file path using the pattern f"{data}_data.pt", but the actual dataset files are named {data}.pt (e.g. cells.pt), with no _data suffix 
# ErrNo : Zero Grad missing in fit.py file
There is zero Grad added to the optimizer 
# ErrNO : Renaming sum to total 
Renamed variable name sum to total to avoid shawdoing 
# Err No: RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x3072 and 2048x1024)
# fix : 
# Err No: RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x3072 and 2048x1024)
# fix : 
The matrix 1 is has a shape of (32*3072) and matrix 2 has a shape of (2048x1024) 
the matrix rule is that the no.of coloums in the matrix 1 should match with the no.of rows in the matrix 2 . Here they are not matching in the model AlexNet. 
So changing the shape of  nn.Linear(2048, 1024) from ALexNet to nn.Linear(3072, 1024) to match the coloum and row of the matrixs . 
# Err No : `RuntimeError: 0D or 1D target tensor expected, multi-target not supported`
train_labels / val_labels were loaded directly from the ".pt" file with shape [N, 1] 
torch.Size([13671, 1]) each label wrapped in its own extra dimension. PyTorch's "CrossEntropyLoss" requires targets to be 1D (shape "[N]", one class index per sample), so it rejected the 2D input rather than guessing the intended shape.
# Fix : 
Wrapped the label extraction in torch.squeeze(...) inside data.py removing the extra dimension so labels go from [N, 1] to :[N]".

# Silent Bug 1: in_channels and num_classes were hardcoded
Problem: AlexNet's __init__ took **kwargs but never pulled in_channels or num_classes out of it. So no matter what the config.json said, the model always built itself with 3 channels and 11 classes. Even when I set NUM_CLASSES to 8 in config, the model still had an 11-class output layer — it just never crashed, so it was easy to miss.
Fix: Added in_channels and num_classes as proper parameters in __init__, so the model now actually uses whatever config.json says instead of ignoring it.

# Silent Bug 2: Activation was Identity instead of ReLU
Problem: train.py was passing activation_str=None into the model, which became nn.Identity() — meaning no activation at all between layers. Without ReLU, the whole stack of conv and linear layers just behaves like one big linear equation, so it can't learn complex patterns. Training accuracy would climb for the first couple epochs and then collapse and get stuck around random-guessing level.
Fix: Set the activation to nn.ReLU(inplace=True) directly in AlexNet's features and classifier.

# Silent Bug 3: in_channels/num_classes fallback values were swapped
Problem: When I first added kwargs.get() fallback defaults, I wrote in_channels defaulting to 11 and num_classes defaulting to 3 — backwards from what they should be. Since config.json always supplied real values, this never actually showed up while running, but it would've silently misconfigured the model if anyone ever built it without passing those arguments.
Fix: Corrected the defaults so in_channels falls back to 3 and num_classes falls back to 11, matching the original hardcoded values.
# VGG 16 errors nd bugs 
RuntimeError: Given groups=1, weight of size [64, 3, 3, 3], expected input[32, 64, 64, 64] to have 3 channels, but got 64 channels instead
fix: VGGBlock conv layers reused original in_channels instead of updating to out_channels after first conv


# Resnet 
TypeError: cross_entropy_loss(): argument 'input' (position 1) must be Tensor, not NoneType
(.venv311) shravanijoshi@Mac Code % 