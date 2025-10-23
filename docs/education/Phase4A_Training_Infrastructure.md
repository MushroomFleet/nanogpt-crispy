# Phase 4A: Training Infrastructure

## Substage Overview

- **Parent Phase:** Phase 4: Training Loop
- **Substage:** A of 3 (Infrastructure)
- **Goal:** Set up the basic training infrastructure and model initialization
- **Prerequisites:** Phase 1-3 completed (setup, model, data ready)
- **Estimated Duration:** 2 hours
- **Key Deliverables:**
  - Complete training script structure (`train.py`)
  - Configuration loading system (`configurator.py`)
  - Data loading functions
  - Model initialization (scratch/resume/pretrained modes)
  - Basic training iteration
  - Simple logging

---

## What We're Building

Before we can train models, we need infrastructure:

1. **Training Script** - Main entry point with configuration
2. **Config Loader** - Flexible parameter overriding
3. **Data Loading** - Efficient batch generation
4. **Model Init** - Three ways to start: scratch, resume, or from GPT-2
5. **Basic Loop** - Simple forward/backward pass

In the next substages (4B & 4C), we'll add optimization, scheduling, and production features.

---

## Training Loop Architecture

```
Training Infrastructure

Command Line
  ↓
Load Config (configurator.py)
  ↓
Initialize:
├── Data (load train/val splits)
├── Model (scratch/resume/pretrained)
├── Optimizer (Phase 4B)
└── Scheduler (Phase 4B)
  ↓
Basic Training Loop:
├── Get batch
├── Forward pass → loss
├── Backward pass → gradients
├── Optimizer step (Phase 4B)
└── Log progress
  ↓
Evaluation & Checkpointing (Phase 4C)
```

---

## Step 1: Training Script Structure

**Purpose:** Create the main training script with configuration  
**Duration:** 30 minutes

### Create train.py

Create `train.py` in the project root:

```python
# train.py
"""
Train a GPT model on a text dataset.

Usage:
    # Single GPU
    python train.py config/train_shakespeare_char.py

    # Multi-GPU (DDP) - Phase 4C
    torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Default config values (can be overridden by config file)
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # If True, exit after first eval
always_save_checkpoint = True  # If True, always save after each eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'

# Data
dataset = 'shakespeare'
gradient_accumulation_steps = 1  # Simulate larger batches (Phase 4B)
batch_size = 12  # Per-GPU batch size
block_size = 1024  # Context length

# Model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # For pretraining 0 is good, for finetuning try 0.1+
bias = False  # Do we use bias inside LayerNorm and Linear layers?

# AdamW optimizer (Phase 4B)
learning_rate = 6e-4  # Max learning rate
max_iters = 600000  # Total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # Clip gradients at this value, or disable if == 0.0

# Learning rate decay (Phase 4B)
decay_lr = True  # Whether to decay the learning rate
warmup_iters = 2000  # How many steps to warm up for
lr_decay_iters = 600000  # Should be ~= max_iters
min_lr = 6e-5  # Minimum learning rate, should be ~= learning_rate/10

# DDP settings (Phase 4C)
backend = 'nccl'  # 'nccl', 'gloo', etc.

# System
device = 'cuda'  # 'cpu', 'cuda', 'cuda:0', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True  # Use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # Overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # Will be useful for logging
# -----------------------------------------------------------------------------

# Various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1  # Is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # This process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # Each process gets a different seed
    # World_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # Single-GPU or CPU run
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# Calculate tokens per iteration
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    print(f"Tokens per iteration: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(1337 + seed_offset)

# Note: float16 data type will automatically use a GradScaler
torch.backends.cuda.matmul.allow_tf32 = True  # Allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # Allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # For later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
```

### Understanding the Setup

**Configuration Variables:**
- Grouped by purpose (I/O, Data, Model, Optimizer, System)
- All can be overridden from config files or command line
- Stored in `config` dict for logging

**DDP Detection:**
```python
ddp = int(os.environ.get('RANK', -1)) != -1
```
- Checks for distributed training environment variables
- Sets up multi-GPU if detected
- Falls back to single GPU/CPU otherwise

**Mixed Precision Setup:**
```python
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
```
- Enables automatic mixed precision
- Uses BF16 if available, else FP16
- 2x speedup typically

---

## Step 2: Configuration Loader

**Purpose:** Flexible config override system  
**Duration:** 10 minutes

### Create configurator.py

```python
# configurator.py
"""
Simple config loader that can override defaults from:
1. Config file (if provided as first argument)
2. Command line arguments (--key=value)
"""

import sys

# Parse command line arguments
for arg in sys.argv[1:]:
    if '=' not in arg:
        # Assume it's a config file
        assert arg.endswith('.py')
        with open(arg) as f:
            print(f"Overriding config with {arg}:")
            exec(f.read())
    else:
        # Assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]

        # Attempt to eval it (e.g. for int, float, bool)
        try:
            attempt = eval(val)
        except (SyntaxError, NameError):
            attempt = val

        # Ensure the key is valid
        assert key in globals(), f"Unknown config key: {key}"

        # Cross-validate types
        default_val = globals()[key]
        assert type(attempt) == type(default_val), f"Type mismatch for {key}: {type(attempt)} vs {type(default_val)}"

        # Override
        print(f"Overriding: {key} = {attempt}")
        globals()[key] = attempt
```

### How It Works

**Loading from file:**
```bash
python train.py config/train_shakespeare_char.py
```
- Executes the config file
- Overrides defaults in train.py

**Command line overrides:**
```bash
python train.py --learning_rate=1e-3 --max_iters=1000
```
- Override specific parameters
- Type-checked against defaults

---

## Step 3: Data Loading

**Purpose:** Load datasets and create batches  
**Duration:** 20 minutes

### Add Data Functions to train.py

Add these functions after the configuration section:

```python
# Data loading
def get_data_path(dataset):
    """Get path to dataset directory."""
    return os.path.join('data', dataset)


def load_dataset_metadata(dataset):
    """Load dataset metadata."""
    data_dir = get_data_path(dataset)
    meta_path = os.path.join(data_dir, 'meta.pkl')

    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
    else:
        # Assume GPT-2 defaults if no meta.pkl
        meta = {'vocab_size': 50304}

    return meta


def load_tokens(dataset, split):
    """Load tokens from binary file."""
    data_dir = get_data_path(dataset)
    bin_path = os.path.join(data_dir, f'{split}.bin')

    if not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"Dataset file not found: {bin_path}\n"
            f"Please run: python data/{dataset}/prepare.py"
        )

    return np.memmap(bin_path, dtype=np.uint16, mode='r')


def get_batch(split, train_data=None, val_data=None):
    """Generate a batch of data."""
    data = train_data if split == 'train' else val_data

    # Sample random positions
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    # Pin memory for faster GPU transfer
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


# Load data
if master_process:
    print(f"Loading dataset: {dataset}")

meta = load_dataset_metadata(dataset)
train_data = load_tokens(dataset, 'train')
val_data = load_tokens(dataset, 'val')

if master_process:
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
```

### Understanding Data Loading

**Memory-Mapped Loading:**
```python
np.memmap(bin_path, dtype=np.uint16, mode='r')
```
- Doesn't load entire file into RAM
- OS loads chunks as needed
- Essential for large datasets

**Pin Memory:**
```python
x = x.pin_memory().to(device, non_blocking=True)
```
- Speeds up CPU → GPU transfers
- Uses page-locked memory
- Significant performance boost

---

## Step 4: Model Initialization

**Purpose:** Create or load the model  
**Duration:** 30 minutes

### Add Model Init to train.py

```python
# Model initialization
if master_process:
    print(f"Initializing model from '{init_from}'")

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)

if init_from == 'scratch':
    # Init a new model from scratch
    if master_process:
        print("Initializing a new model from scratch")

    # Determine vocab size
    if meta.get('vocab_size') is not None:
        model_args['vocab_size'] = meta['vocab_size']
    else:
        if master_process:
            print("Defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = 50304

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    # Resume training from a checkpoint
    if master_process:
        print(f"Resuming training from {out_dir}")

    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)

    checkpoint_model_args = checkpoint['model_args']

    # Force these config attributes to be equal otherwise we can't even resume training
    # The rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

    # Create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']

    # Fix the keys of the state dictionary
    # (Honestly not sure why this is needed, but it is for DDP models)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    iter_num = checkpoint.get('iter_num', 0)
    best_val_loss = checkpoint.get('best_val_loss', 1e9)

elif init_from.startswith('gpt2'):
    # Initialize from OpenAI GPT-2 weights
    if master_process:
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")

    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)

    # Read off and override the GPT sizing model args from the model config
    model_args['n_layer'] = model.config.n_layer
    model_args['n_head'] = model.config.n_head
    model_args['n_embd'] = model.config.n_embd
    model_args['block_size'] = model.config.block_size
    model_args['bias'] = model.config.bias
    model_args['vocab_size'] = model.config.vocab_size

    # Crop block size if needed
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size

# Move model to device
model.to(device)

# Initialize a GradScaler for mixed precision training
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Compile the model (PyTorch 2.0+)
if compile:
    if master_process:
        print("Compiling the model... (takes ~1 minute)")
    unoptimized_model = model
    model = torch.compile(model)  # Requires PyTorch 2.0

# Wrap model in DDP container (Phase 4C)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
```

### Three Initialization Modes

**1. From Scratch (`init_from='scratch'`):**
- Random initialization
- For training from scratch
- Most compute intensive

**2. Resume (`init_from='resume'`):**
- Continue from checkpoint
- Loads model + optimizer + iteration state
- For interrupted training

**3. Pretrained (`init_from='gpt2'`):**
- Start from GPT-2 weights
- Fine-tune on custom data
- Much faster than training from scratch
- Options: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

---

## Step 5: Basic Training Iteration

**Purpose:** Simple forward/backward pass  
**Duration:** 30 minutes

### Add Basic Training Loop

Add to train.py:

```python
# Initialize tracking
iter_num = 0 if init_from == 'scratch' or init_from.startswith('gpt2') else checkpoint['iter_num']
best_val_loss = 1e9 if init_from == 'scratch' or init_from.startswith('gpt2') else checkpoint['best_val_loss']

# Optimizer will be added in Phase 4B
# For now, we'll just do forward passes to test everything works

print(f"\nStarting training from iteration {iter_num}")
print(f"Training for {max_iters} iterations")
print("="*60)

# Simple training loop (no optimization yet - that's Phase 4B)
for iteration in range(5):  # Just 5 iterations to test
    print(f"\nIteration {iteration}:")
    
    # Get batch
    X, Y = get_batch('train', train_data, val_data)
    print(f"  Batch shape: X={X.shape}, Y={Y.shape}")
    
    # Forward pass
    with ctx:
        logits, loss = model(X, Y)
    
    print(f"  Loss: {loss.item():.4f}")
    
    # In Phase 4B, we'll add:
    # - Backward pass
    # - Optimizer step
    # - Gradient accumulation
    # - Gradient clipping

print("\n" + "="*60)
print("Basic training iteration test complete!")
print("In Phase 4B, we'll add optimization and scheduling.")
```

---

## Step 6: Test Configuration

**Purpose:** Create test config for quick validation  
**Duration:** 10 minutes

### Create config/test_config.py

Create a simple config for testing:

```python
# config/test_config.py
"""
Test configuration for Phase 4A.
Very small model, few iterations, just to verify everything works.
"""

# I/O
out_dir = 'out-test'
eval_interval = 10
log_interval = 1
eval_iters = 5
always_save_checkpoint = False

# Data
dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 4
block_size = 64

# Model - Tiny for fast testing
n_layer = 2
n_head = 2
n_embd = 128
dropout = 0.0

# Training
learning_rate = 1e-3
max_iters = 20
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR decay
decay_lr = False

# System
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float32'
compile = False  # Disable for testing
```

### Test It

```bash
python train.py config/test_config.py
```

### Expected Output

```
Overriding config with config/test_config.py:
Tokens per iteration: 256
Loading dataset: shakespeare
Train tokens: 304,222
Val tokens: 33,803
Initializing model from 'scratch'
Initializing a new model from scratch
Number of parameters: 0.75M

Starting training from iteration 0
Training for 20 iterations
============================================================

Iteration 0:
  Batch shape: X=torch.Size([4, 64]), Y=torch.Size([4, 64])
  Loss: 10.8234

Iteration 1:
  Batch shape: X=torch.Size([4, 64]), Y=torch.Size([4, 64])
  Loss: 10.7123

...

============================================================
Basic training iteration test complete!
In Phase 4B, we'll add optimization and scheduling.
```

---

## Troubleshooting

### Issue: "Dataset file not found"

**Problem:** Missing prepared data

**Solution:**
```bash
python data/shakespeare/prepare.py
```

### Issue: CUDA out of memory

**Solutions:**
1. Reduce `batch_size`
2. Reduce `block_size`
3. Use smaller model (fewer `n_layer`, smaller `n_embd`)

### Issue: "Config key not found"

**Problem:** Trying to override non-existent parameter

**Solution:**
- Check parameter name spelling
- Ensure it's defined in train.py defaults
- Add to `config_keys` list if needed

---

## Phase 4A Complete!

### What You've Built

✅ **Training Infrastructure**:
- Complete training script structure (`train.py`)
- Flexible configuration loading (`configurator.py`)
- Data loading with memory-mapping
- Three model initialization modes
- Basic training iteration (forward pass)
- Mixed precision setup
- DDP detection (multi-GPU ready)

### Key Files Created

- `train.py`: Main training script (~200 lines so far)
- `configurator.py`: Config override system
- `config/test_config.py`: Test configuration

### What's Missing (Next Substages)

**Phase 4B will add:**
- AdamW optimizer with proper weight decay
- Learning rate scheduling (warmup + cosine)
- Backward pass and gradient computation
- Gradient accumulation
- Gradient clipping
- MFU (Model FLOPs Utilization) estimation

**Phase 4C will add:**
- Evaluation loop
- Checkpoint saving/loading
- Multi-GPU (DDP) training
- Complete training loop
- Progress monitoring

---

## Next Substage

**Proceed to Phase 4B: Optimization & Scheduling**

Now that we have the infrastructure, we'll add:
- Optimizer configuration
- Learning rate scheduling
- Gradient operations
- Performance metrics

**Estimated time:** 2 hours

This is where the model actually learns!

---

**Phase 4A Character Count:** ~19,500 characters
