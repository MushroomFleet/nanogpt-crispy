# Phase 1: Project Setup and Configuration

## Phase Overview

**Goal:** Establish a clean development environment and project structure for NanoGPT implementation

**Prerequisites:**
- Python 3.8+ installed
- CUDA-capable GPU (optional for development, required for training)
- Basic knowledge of Python and command line

**Estimated Duration:** 1-2 hours

**Key Deliverables:**
- Working Python environment with PyTorch
- Organized project directory structure
- Configuration management system
- Version control repository
- Development utilities and helpers

---

## Step 1: Environment Setup

**Purpose:** Create an isolated Python environment with all necessary dependencies
**Duration:** 15-20 minutes

### Instructions

1. **Create project directory**
```bash
mkdir nanogpt
cd nanogpt
```

2. **Initialize Git repository**
```bash
git init
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

3. **Create Python virtual environment**
```bash
# Using venv (built-in)
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

4. **Create requirements.txt**

```txt
# requirements.txt
torch>=2.0.0
numpy>=1.24.0
tiktoken>=0.4.0
```

5. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Verification
- [ ] Virtual environment activated
- [ ] PyTorch installed successfully
- [ ] CUDA detected (if GPU available)
- [ ] Git repository initialized

---

## Step 2: Project Structure

**Purpose:** Create organized directory structure for code, data, and outputs
**Duration:** 5 minutes

### Instructions

1. **Create directory structure**
```bash
mkdir -p data/shakespeare
mkdir -p data/openwebtext
mkdir -p config
mkdir -p out
mkdir -p tests
```

2. **Create initial files**
```bash
# Core files (we'll implement these in later phases)
touch model.py
touch train.py
touch sample.py

# Config files
touch config/train_shakespeare_char.py
touch config/train_gpt2.py
touch config/finetune_shakespeare.py

# Data preparation
touch data/shakespeare/prepare.py

# Tests
touch tests/test_model.py
touch tests/test_training.py

# Documentation
touch README.md
```

3. **Create .gitignore**

```gitignore
# .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# PyTorch
*.pt
*.pth
*.ckpt

# Data files
*.bin
*.txt
!requirements.txt
data/*/train.bin
data/*/val.bin

# Outputs
out/
wandb/
logs/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Temporary
*.log
.cache/
```

### Directory Structure Result
```
nanogpt/
├── venv/                 # Virtual environment
├── model.py              # GPT model (Phase 2)
├── train.py              # Training loop (Phase 4)
├── sample.py             # Generation (Phase 5)
├── requirements.txt
├── README.md
├── .gitignore
├── config/
│   ├── train_shakespeare_char.py
│   ├── train_gpt2.py
│   └── finetune_shakespeare.py
├── data/
│   ├── shakespeare/
│   │   └── prepare.py
│   └── openwebtext/
├── out/                  # Training checkpoints
└── tests/
    ├── test_model.py
    └── test_training.py
```

### Verification
- [ ] All directories created
- [ ] Core files exist (empty for now)
- [ ] .gitignore configured
- [ ] Project structure matches expected layout

---

## Step 3: Configuration System

**Purpose:** Build a flexible configuration system for experiments
**Duration:** 20-30 minutes

### Instructions

NanoGPT uses a simple but powerful configuration pattern: Python files as configs. This allows for dynamic configuration while maintaining simplicity.

1. **Create base configuration** (`config/train_shakespeare_char.py`)

```python
# config/train_shakespeare_char.py
"""
Train a character-level GPT on Shakespeare text.
This is the simplest possible configuration for quick experimentation.
"""

# I/O
out_dir = 'out-shakespeare-char'
eval_interval = 250         # Evaluate every N iterations
log_interval = 10           # Log every N iterations
eval_iters = 200            # Number of iterations for evaluation
always_save_checkpoint = True

# Data
dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256            # Context length

# Model - Small model for fast training
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# Optimizer
learning_rate = 1e-3
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0             # Clip gradients at this value

# Learning rate decay
decay_lr = True
warmup_iters = 100          # Linear warmup
lr_decay_iters = 5000       # Cosine decay to min_lr
min_lr = 1e-4               # Minimum learning rate

# System
device = 'cuda'             # 'cuda', 'cpu', 'mps' (for Mac)
dtype = 'float16'           # 'float32', 'float16', 'bfloat16'
compile = False             # Use PyTorch 2.0 compile (set to True if PyTorch 2.0+)
```

2. **Create GPT-2 reproduction config** (`config/train_gpt2.py`)

```python
# config/train_gpt2.py
"""
Train a GPT-2 (124M) from scratch on OpenWebText.
This matches the GPT-2 paper configuration exactly.

Requires 8x A100 40GB GPUs and ~4 days of training.
Expected validation loss: ~2.85
"""

# I/O
out_dir = 'out-gpt2'
eval_interval = 1000
log_interval = 1
eval_iters = 200
always_save_checkpoint = False  # Only save when val loss improves

# Data
dataset = 'openwebtext'
gradient_accumulation_steps = 40  # Effective batch size = 12 * 40 = 480
batch_size = 12                   # Per-GPU batch size
block_size = 1024                 # GPT-2 context length

# Model - GPT-2 (124M)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0           # No dropout (GPT-2 paper)
bias = False            # GPT-2 doesn't use bias

# Optimizer - Matches GPT-2 paper
learning_rate = 6e-4
max_iters = 600000      # ~300B tokens
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# DDP settings (multi-GPU)
backend = 'nccl'        # NCCL for GPUs, gloo for CPU

# System
device = 'cuda'
dtype = 'bfloat16'      # Use bfloat16 if available (better than float16)
compile = True          # PyTorch 2.0 compilation (2x speedup)
```

3. **Create fine-tuning config** (`config/finetune_shakespeare.py`)

```python
# config/finetune_shakespeare.py
"""
Fine-tune GPT-2 (1.5B) on Shakespeare dataset.
Downloads pretrained GPT-2 and adapts it to Shakespeare's writing style.

This is VERY fast (~5 minutes on a single GPU) and produces impressive results.
"""

# I/O
out_dir = 'out-shakespeare-finetuned'
eval_interval = 5
log_interval = 1
eval_iters = 20
always_save_checkpoint = False

# Wandb logging (optional)
wandb_log = False
wandb_project = 'shakespeare'
wandb_run_name = 'gpt2-finetuned'

# Data
dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 1              # Very small batch for fine-tuning
block_size = 1024

# Model - Start from GPT-2 checkpoint
init_from = 'gpt2-xl'       # 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

# Fine-tuning hyperparameters
learning_rate = 3e-5        # Much smaller than pre-training
max_iters = 20              # Very few iterations needed
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 0            # No warmup for fine-tuning
lr_decay_iters = 20
min_lr = 3e-5

# System
device = 'cuda'
dtype = 'bfloat16'
compile = False             # Can't compile when loading from HF checkpoint
```

4. **Create config loader utility**

Add this to the top of your `train.py` (we'll create full train.py in Phase 4):

```python
# train.py (partial - just config loading)
import os
import sys
from contextlib import nullcontext

# Configuration loading
config_keys = [
    'out_dir', 'eval_interval', 'log_interval', 'eval_iters',
    'always_save_checkpoint', 'dataset', 'gradient_accumulation_steps',
    'batch_size', 'block_size', 'n_layer', 'n_head', 'n_embd',
    'dropout', 'bias', 'learning_rate', 'max_iters', 'weight_decay',
    'beta1', 'beta2', 'grad_clip', 'decay_lr', 'warmup_iters',
    'lr_decay_iters', 'min_lr', 'device', 'dtype', 'compile'
]

def load_config(config_path):
    """
    Load configuration from a Python file.

    Args:
        config_path: Path to config file (e.g., 'config/train_gpt2.py')

    Returns:
        dict: Configuration dictionary
    """
    config = {}

    if config_path:
        # Execute config file in isolated namespace
        with open(config_path, 'r') as f:
            exec(f.read(), config)

    # Filter to only known config keys
    config = {k: v for k, v in config.items() if k in config_keys}

    return config

# Usage example:
# python train.py config/train_shakespeare_char.py
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    config = load_config(config_file)
    print(f"Loaded config from {config_file}")
    for k, v in config.items():
        print(f"  {k}: {v}")
```

### Configuration Pattern Benefits

**Advantages:**
- Pure Python: Full expressiveness (conditionals, computations, etc.)
- No YAML/JSON parsing complexity
- Type safety and IDE autocomplete
- Easy to version control and diff
- Can compute derived values

**Example of dynamic configuration:**
```python
# config/custom.py

# Compute effective batch size
batch_size = 12
gradient_accumulation_steps = 40
effective_batch_size = batch_size * gradient_accumulation_steps  # 480

# Conditional configuration
import torch
if torch.cuda.is_available():
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
else:
    device = 'cpu'
    dtype = 'float32'

# Derived values
max_iters = 600000
eval_interval = max_iters // 100  # Evaluate 100 times during training
```

### Verification
- [ ] Three config files created
- [ ] Config loader function implemented
- [ ] Can run `python train.py config/train_shakespeare_char.py` without errors
- [ ] Configuration values print correctly

---

## Step 4: Utility Functions

**Purpose:** Create helper functions used throughout the project
**Duration:** 15-20 minutes

### Instructions

1. **Create utils.py** (optional but recommended)

```python
# utils.py
"""
Utility functions for NanoGPT implementation.
"""

import os
import pickle
import numpy as np
import torch

def set_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_lr(it, config):
    """
    Calculate learning rate with warmup and cosine decay.

    Args:
        it: Current iteration
        config: Training configuration dict

    Returns:
        float: Learning rate for current iteration
    """
    warmup_iters = config.get('warmup_iters', 0)
    lr_decay_iters = config.get('lr_decay_iters', config['max_iters'])
    learning_rate = config['learning_rate']
    min_lr = config.get('min_lr', learning_rate * 0.1)

    # 1) Linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    # 2) Constant (no decay)
    if not config.get('decay_lr', True):
        return learning_rate

    # 3) Cosine decay to min_lr
    if it > lr_decay_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))  # Cosine from 1 to 0
    return min_lr + coeff * (learning_rate - min_lr)

def count_parameters(model):
    """
    Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_mfu(model, fwdbwd_per_iter, dt, config):
    """
    Estimate Model FLOPs Utilization (MFU).

    This measures what percentage of GPU's theoretical FLOP/s we're achieving.
    GPT-3 paper reported 21% MFU on A100 GPUs.

    Args:
        model: GPT model
        fwdbwd_per_iter: Forward/backward passes per iteration
        dt: Time per iteration (seconds)
        config: Model configuration

    Returns:
        float: MFU estimate (0.0 to 1.0)
    """
    # Estimate FLOPs per token
    N = count_parameters(model)
    L, H, Q, T = config['n_layer'], config['n_head'], config['n_embd']//config['n_head'], config['block_size']

    # Per-token FLOPs (from Kaplan et al. 2020)
    # Forward pass: 6N (6 ops per param: matmul + add)
    # Backward pass: 2 * forward (gradients + weight updates)
    flops_per_token = 6 * N
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

    # FLOPs achieved
    flops_achieved = flops_per_iter / dt

    # A100 theoretical peak: 312 TFLOP/s (BF16)
    # Adjust for your GPU
    flops_promised = 312e12  # A100

    mfu = flops_achieved / flops_promised
    return mfu

def save_checkpoint(model, optimizer, iter_num, config, val_loss, filepath):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        iter_num: Current iteration
        config: Training config
        val_loss: Validation loss
        filepath: Checkpoint save path
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num,
        'config': config,
        'val_loss': val_loss,
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint to {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)

    Returns:
        dict: Checkpoint metadata (iter_num, config, val_loss)
    """
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['model'])

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return {
        'iter_num': checkpoint.get('iter_num', 0),
        'config': checkpoint.get('config', {}),
        'val_loss': checkpoint.get('val_loss', float('inf')),
    }

def get_batch(data, block_size, batch_size, device):
    """
    Generate a random batch from data.

    Args:
        data: numpy array of token IDs
        block_size: Context length
        batch_size: Batch size
        device: torch device

    Returns:
        tuple: (x, y) where x is input and y is target
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y
```

2. **Test utilities**

```python
# Test utils
if __name__ == '__main__':
    # Test learning rate schedule
    config = {
        'learning_rate': 6e-4,
        'warmup_iters': 100,
        'lr_decay_iters': 1000,
        'min_lr': 6e-5,
        'max_iters': 1000,
        'decay_lr': True,
    }

    print("Learning rate schedule:")
    for it in [0, 50, 100, 500, 999, 1000]:
        lr = get_lr(it, config)
        print(f"  Iter {it:4d}: {lr:.2e}")

    # Expected output:
    # Iter    0: 0.00e+00  (warmup start)
    # Iter   50: 3.00e-04  (warmup middle)
    # Iter  100: 6.00e-04  (warmup end, peak LR)
    # Iter  500: 3.30e-04  (decay middle)
    # Iter  999: 6.00e-05  (decay end, min LR)
    # Iter 1000: 6.00e-05  (after decay)
```

### Verification
- [ ] utils.py created with all helper functions
- [ ] Learning rate schedule works correctly
- [ ] Functions have proper docstrings
- [ ] No syntax errors when importing

---

## Step 5: Initial README

**Purpose:** Document the project setup and usage
**Duration:** 10 minutes

### Instructions

Create `README.md`:

```markdown
# NanoGPT

A minimal, educational implementation of GPT (Generative Pre-trained Transformer) in PyTorch.

## Overview

NanoGPT is a clean, readable implementation of the GPT architecture that can:
- Reproduce GPT-2 (124M) from scratch
- Fine-tune pretrained GPT-2 models
- Train character-level models on small datasets
- Serve as an educational resource for understanding transformers

**Total code**: ~600 lines (model.py + train.py)

## Features

- Pure PyTorch implementation (no framework dependencies)
- Configurable model sizes (124M to 1.5B+ parameters)
- Multi-GPU training with PyTorch DDP
- PyTorch 2.0 compilation support (2x speedup)
- Mixed precision training (FP16/BF16)
- GPT-2 checkpoint compatibility

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd nanogpt

# Create environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train on Shakespeare (5 minutes)

```bash
# Prepare data
python data/shakespeare/prepare.py

# Train small character-level model
python train.py config/train_shakespeare_char.py

# Generate text
python sample.py --out_dir=out-shakespeare-char
```

### 2. Fine-tune GPT-2

```bash
# Prepare Shakespeare data
python data/shakespeare/prepare.py

# Fine-tune GPT-2 XL (1.5B) on Shakespeare
python train.py config/finetune_shakespeare.py

# Generate Shakespeare-style text
python sample.py --out_dir=out-shakespeare-finetuned
```

### 3. Reproduce GPT-2 (multi-day, 8× A100)

```bash
# Prepare OpenWebText dataset
python data/openwebtext/prepare.py

# Train on 8 GPUs
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

## Project Structure

```
nanogpt/
├── model.py              # GPT architecture (~300 lines)
├── train.py              # Training loop (~300 lines)
├── sample.py             # Text generation
├── utils.py              # Helper functions
├── config/               # Training configurations
├── data/                 # Datasets
├── out/                  # Checkpoints
└── tests/                # Unit tests
```

## Configuration

See `config/` directory for example configurations:
- `train_shakespeare_char.py`: Fast character-level training
- `train_gpt2.py`: GPT-2 (124M) reproduction
- `finetune_shakespeare.py`: GPT-2 fine-tuning

## Requirements

**Minimum**:
- Python 3.8+
- Single GPU with 8GB+ VRAM
- PyTorch 2.0+

**Recommended** (GPT-2 reproduction):
- 8× A100 40GB GPUs
- PyTorch 2.0+ with CUDA 11.8+

## License

MIT

## References

- Original NanoGPT: https://github.com/karpathy/nanoGPT
- GPT-2 Paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Attention Paper: https://arxiv.org/abs/1706.03762
```

### Verification
- [ ] README.md created
- [ ] Installation instructions clear
- [ ] Quick start examples included
- [ ] Project structure documented

---

## Step 6: Initial Commit

**Purpose:** Save initial project setup to version control
**Duration:** 5 minutes

### Instructions

1. **Check status**
```bash
git status
```

2. **Add files**
```bash
git add .
git status  # Verify files to be committed
```

3. **Create initial commit**
```bash
git commit -m "Initial project setup

- Python environment with PyTorch 2.0+
- Project directory structure
- Configuration system (Shakespeare, GPT-2, fine-tuning)
- Utility functions (LR schedule, checkpointing, etc.)
- README with installation and usage instructions
"
```

4. **Verify commit**
```bash
git log --oneline
git show HEAD  # View commit details
```

### Verification
- [ ] All files committed
- [ ] Commit message is descriptive
- [ ] Git log shows initial commit

---

## Testing Procedures

### Environment Test

Run this Python script to verify environment setup:

```python
# test_environment.py
import sys
import torch
import numpy as np
import tiktoken

def test_environment():
    print("=== Environment Test ===\n")

    # Python version
    print(f"Python: {sys.version}")
    assert sys.version_info >= (3, 8), "Python 3.8+ required"
    print("✓ Python version OK\n")

    # PyTorch
    print(f"PyTorch: {torch.__version__}")
    assert torch.__version__ >= "2.0", "PyTorch 2.0+ recommended"
    print("✓ PyTorch version OK\n")

    # CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("✓ CUDA check complete\n")

    # NumPy
    print(f"NumPy: {np.__version__}")
    print("✓ NumPy OK\n")

    # tiktoken
    enc = tiktoken.get_encoding("gpt2")
    test_text = "Hello, world!"
    tokens = enc.encode(test_text)
    decoded = enc.decode(tokens)
    print(f"tiktoken test: '{test_text}' -> {tokens} -> '{decoded}'")
    assert decoded == test_text, "tiktoken roundtrip failed"
    print("✓ tiktoken OK\n")

    # Simple PyTorch operations
    x = torch.randn(2, 3)
    y = torch.randn(3, 2)
    z = torch.matmul(x, y)
    print(f"PyTorch matmul test: ({x.shape} @ {y.shape} = {z.shape})")
    print("✓ PyTorch operations OK\n")

    # GPU test (if available)
    if cuda_available:
        x_gpu = torch.randn(1000, 1000, device='cuda')
        y_gpu = torch.randn(1000, 1000, device='cuda')
        z_gpu = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()
        print(f"GPU matmul test: {z_gpu.shape} on {z_gpu.device}")
        print("✓ GPU operations OK\n")

    print("=== All tests passed! ===")

if __name__ == '__main__':
    test_environment()
```

Run test:
```bash
python test_environment.py
```

Expected output:
```
=== Environment Test ===

Python: 3.12.x ...
✓ Python version OK

PyTorch: 2.0.x
✓ PyTorch version OK

CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce RTX 4090
GPU memory: 24.0 GB
✓ CUDA check complete

NumPy: 1.24.x
✓ NumPy OK

tiktoken test: 'Hello, world!' -> [15496, 11, 995, 0] -> 'Hello, world!'
✓ tiktoken OK

PyTorch matmul test: (torch.Size([2, 3]) @ torch.Size([3, 2]) = torch.Size([2, 2]))
✓ PyTorch operations OK

GPU matmul test: torch.Size([1000, 1000]) on cuda:0
✓ GPU operations OK

=== All tests passed! ===
```

---

## Troubleshooting

### Issue: CUDA not available

**Symptoms**: `torch.cuda.is_available()` returns `False`

**Solutions**:
1. Install CUDA-enabled PyTorch:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. Verify NVIDIA driver:
   ```bash
   nvidia-smi
   ```

3. For development without GPU:
   - CPU-only PyTorch works fine for testing
   - Training will be very slow
   - Consider using Google Colab for free GPU access

### Issue: Import errors

**Symptoms**: `ModuleNotFoundError: No module named 'X'`

**Solutions**:
1. Ensure virtual environment is activated:
   ```bash
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Reinstall requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Check Python path:
   ```bash
   which python  # Should point to venv/bin/python
   ```

### Issue: PyTorch version mismatch

**Symptoms**: Features not available, unexpected behavior

**Solutions**:
1. Upgrade PyTorch:
   ```bash
   pip install --upgrade torch
   ```

2. Check version:
   ```python
   import torch
   print(torch.__version__)  # Should be 2.0.0 or higher
   ```

---

## Next Steps

✅ **Phase 1 Complete!** You now have:
- Working development environment
- Organized project structure
- Configuration system
- Utility functions
- Version control

**Proceed to Phase 2**: Core Model Architecture
- Implement GPT model components
- Build self-attention mechanism
- Create transformer blocks
- Test forward passes

**Estimated time for Phase 2**: 4-6 hours

---

**Phase 1 Character Count**: ~24,500 characters
