# Phase 4: Training Loop

## Phase Overview

**Goal:** Implement complete training infrastructure with optimization and checkpointing

**Prerequisites:**
- Phase 1-3 completed (environment, model, data)
- Understanding of gradient descent and backpropagation
- Familiarity with PyTorch training loops

**Estimated Duration:** 5-7 hours

**Key Deliverables:**
- Complete `train.py` (~300 lines)
- AdamW optimizer with proper weight decay
- Learning rate scheduling (warmup + cosine decay)
- Training loop with gradient accumulation
- Mixed precision training (FP16/BF16)
- Loss computation and logging
- Evaluation loop
- Checkpoint saving/loading
- Distributed Data Parallel (DDP) support
- Training metrics and progress tracking

---

## Understanding the Training Loop

```
Training Loop Architecture

Initialize:
├── Load config
├── Create model
├── Load data
├── Setup optimizer
└── Setup learning rate scheduler

Training Loop (for each iteration):
├── 1. Get batch from training data
├── 2. Forward pass → compute loss
├── 3. Backward pass → compute gradients
├── 4. (Optional) Gradient accumulation
├── 5. Gradient clipping
├── 6. Optimizer step → update weights
├── 7. Update learning rate
└── 8. Log metrics

Evaluation Loop (periodic):
├── 1. Switch to eval mode
├── 2. Compute validation loss
├── 3. Save checkpoint if best
└── 4. Switch back to train mode

Checkpointing:
├── Save model state
├── Save optimizer state
├── Save training iteration
└── Save configuration
```

---

## Step 1: Training Script Setup

**Purpose:** Create the main training script structure
**Duration:** 30 minutes

### Create train.py

```python
# train.py
"""
Train a GPT model on a text dataset.

Usage:
    # Single GPU
    python train.py config/train_shakespeare_char.py

    # Multi-GPU (DDP)
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
gradient_accumulation_steps = 1  # Simulate larger batches
batch_size = 12  # Per-GPU batch size
block_size = 1024  # Context length

# Model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # For pretraining 0 is good, for finetuning try 0.1+
bias = False  # Do we use bias inside LayerNorm and Linear layers?

# AdamW optimizer
learning_rate = 6e-4  # Max learning rate
max_iters = 600000  # Total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # Clip gradients at this value, or disable if == 0.0

# Learning rate decay
decay_lr = True  # Whether to decay the learning rate
warmup_iters = 2000  # How many steps to warm up for
lr_decay_iters = 600000  # Should be ~= max_iters
min_lr = 6e-5  # Minimum learning rate, should be ~= learning_rate/10

# DDP settings
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

### Create configurator.py

This small utility loads configuration from command line or config files:

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

### Verification
- [ ] train.py created with imports and config
- [ ] configurator.py created
- [ ] DDP setup included
- [ ] Config loading works

---

## Step 2: Data Loading Implementation

**Purpose:** Add data loading functions to train.py
**Duration:** 20 minutes

Add these functions to `train.py`:

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

### Verification
- [ ] Data loading functions added
- [ ] Memory-mapped loading works
- [ ] Batch generation works

---

## Step 3: Model Initialization

**Purpose:** Create or load the model
**Duration:** 30 minutes

Add to `train.py`:

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
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

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

# Wrap model in DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
```

### Explanation

**Three initialization modes:**

1. **'scratch'**: Train from random initialization
   - Used for training from scratch on custom data
   - Requires most compute time

2. **'resume'**: Continue from checkpoint
   - Resume interrupted training
   - Loads model, optimizer, and iteration state

3. **'gpt2*'**: Start from pretrained GPT-2
   - Fine-tune on custom dataset
   - Much faster than training from scratch
   - Options: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

**Model compilation:**
```python
model = torch.compile(model)
```
- PyTorch 2.0 feature
- ~2x speedup on training
- First iteration is slow (compilation overhead)
- Subsequent iterations are fast

**DDP (Distributed Data Parallel):**
- Wraps model for multi-GPU training
- Synchronizes gradients across GPUs
- Each GPU processes different batch
- Effective batch size = batch_size * num_gpus

### Verification
- [ ] Model initialization added
- [ ] All three init modes supported
- [ ] Compilation optional
- [ ] DDP support included

---

## Step 4: Optimizer Setup

**Purpose:** Configure AdamW optimizer with proper weight decay
**Duration:** 15 minutes

Add to `train.py`:

```python
# Optimizer
optimizer = model.module.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type) if ddp else model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # Free up memory

# Compile optimizer (PyTorch 2.0+)
# Note: Currently disabled as it can cause issues
# if compile:
#     optimizer = torch.compile(optimizer)
```

### Learning Rate Scheduling

Add this function to `train.py`:

```python
def get_lr(it):
    """
    Learning rate schedule with linear warmup and cosine decay.

    1) Linear warmup for warmup_iters steps
    2) If it > lr_decay_iters, return min learning rate
    3) In between, use cosine decay down to min learning rate
    """
    # 1) Linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    # 2) If beyond decay, return min
    if it > lr_decay_iters:
        return min_lr

    # 3) Cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)
```

### Explanation

**AdamW Optimizer:**
- Adam with proper weight decay (decoupled)
- Better than Adam for transformers
- Weight decay only on weight matrices, not biases/LayerNorm

**Learning Rate Schedule:**
```
learning_rate (6e-4)
    ↑
    |     /------- Cosine decay -------\
    |    /                              \
    |   /                                \_____ min_lr (6e-5)
    |  /
    | /
    |/_______ Linear warmup
    0         warmup      ...         decay_end
        (2000 iters)            (600k iters)
```

**Why warmup?**
- Prevents instability at start of training
- Gradients are large and noisy initially
- Warmup gives model time to stabilize

**Why cosine decay?**
- Smooth decrease in learning rate
- Helps model converge to better minima
- Standard in transformer training

### Verification
- [ ] Optimizer setup added
- [ ] Learning rate scheduling implemented
- [ ] Optimizer state loading works

---

## Step 5: Evaluation Function

**Purpose:** Implement validation loss computation
**Duration:** 20 minutes

Add to `train.py`:

```python
@torch.no_grad()
def estimate_loss():
    """
    Estimate train and val loss by averaging over multiple batches.

    This gives us a less noisy estimate than single-batch evaluation.
    """
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out
```

### Explanation

**Why estimate over multiple batches?**
- Single batch is noisy
- Average over `eval_iters` batches (e.g., 200)
- Gives more stable validation metrics

**`@torch.no_grad()` decorator:**
- Disables gradient computation
- Saves memory
- Speeds up evaluation

**`model.eval()` and `model.train()`:**
- `eval()`: Disables dropout, sets model to evaluation mode
- `train()`: Enables dropout, sets model to training mode

### Verification
- [ ] estimate_loss function added
- [ ] Evaluates on both train and val
- [ ] No gradient computation during eval

---

## Step 6: Main Training Loop

**Purpose:** Implement the core training loop
**Duration:** 60-90 minutes

Add to `train.py`:

```python
# Training loop
if master_process:
    print(f"Training for {max_iters} iterations")
    print(f"Effective batch size: {tokens_per_iter:,} tokens")

# Initialize tracking variables
iter_num = 0
best_val_loss = 1e9
t0 = time.time()
local_iter_num = 0  # Number of iterations on this process
running_mfu = -1.0

# Get model for estimating FLOPs
raw_model = model.module if ddp else model

while True:
    # Determine learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    # Forward backward update, with optional gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # In DDP training we only need to sync gradients at the last micro step
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx:
            X, Y = get_batch('train', train_data, val_data)
            logits, loss = model(X, Y)
            # Scale the loss to account for gradient accumulation
            loss = loss / gradient_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

    # Clip gradients
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Step the optimizer and scaler
    scaler.step(optimizer)
    scaler.update()

    # Flush the gradients
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        # Get loss as float. Note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps

        # Estimate MFU (model FLOPs utilization)
        if local_iter_num >= 5:  # Let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu

        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    # Termination conditions
    if iter_num > max_iters:
        break

# Clean up DDP
if ddp:
    destroy_process_group()
```

### Explanation

**Gradient Accumulation:**
```python
for micro_step in range(gradient_accumulation_steps):
    loss = loss / gradient_accumulation_steps
    loss.backward()
optimizer.step()
```
- Simulates larger batch size
- Useful when GPU memory is limited
- `gradient_accumulation_steps=4` with `batch_size=12` → effective batch size of 48
- Accumulates gradients over multiple forward/backward passes

**Mixed Precision Training:**
```python
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    logits, loss = model(X, Y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
- Uses FP16 or BF16 instead of FP32
- 2x speedup and 2x less memory
- GradScaler prevents underflow

**Gradient Clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```
- Prevents exploding gradients
- Clips gradient norm to max value (e.g., 1.0)
- Essential for stable training

**Model FLOPs Utilization (MFU):**
- Measures efficiency: % of GPU's theoretical FLOP/s achieved
- GPT-3 paper reported 21% MFU on A100s
- Higher is better (means less wasted compute)

### Verification
- [ ] Training loop implemented
- [ ] Gradient accumulation works
- [ ] Mixed precision supported
- [ ] Gradient clipping included
- [ ] Logging shows progress
- [ ] Checkpointing saves state

---

## Step 7: Model FLOPs Estimation

**Purpose:** Add MFU calculation to model
**Duration:** 15 minutes

Add this method to the `GPT` class in `model.py`:

```python
def estimate_mfu(self, fwdbwd_per_iter, dt):
    """
    Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.

    Args:
        fwdbwd_per_iter: number of forward-backward passes per iteration
        dt: time per iteration in seconds

    Returns:
        float: MFU as a fraction of peak FLOPS
    """
    # First estimate the number of flops we do per iteration
    # See PaLM paper Appendix B: https://arxiv.org/abs/2204.02311
    N = self.get_num_params()
    cfg = self.config
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size

    flops_per_token = 6*N + 12*L*H*Q*T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

    # Express our flops throughput as ratio of A100 bfloat16 peak flops
    flops_achieved = flops_per_iter * (1.0/dt)  # per second
    flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
    mfu = flops_achieved / flops_promised

    return mfu
```

### Explanation

**FLOP Calculation:**
- Forward pass: ~6N FLOPs (N = number of parameters)
- Backward pass: ~2 × forward (gradient computation + weight updates)
- Attention: ~12*L*H*Q*T FLOPs (L=layers, H=heads, Q=head_dim, T=sequence_length)

**A100 Peak Performance:**
- 312 TFLOP/s for BF16 operations
- 19.5 TFLOP/s for FP32 operations
- MFU measures what percentage we achieve

**Typical MFU values:**
- 10-15%: Poorly optimized
- 15-25%: Well-optimized (GPT-3 achieved ~21%)
- 25-40%: Excellent (with flash attention and other optimizations)
- >40%: State-of-the-art optimization

### Verification
- [ ] estimate_mfu method added to GPT class
- [ ] FLOP calculation included
- [ ] MFU displayed during training

---

## Step 8: Testing the Training Loop

**Purpose:** Verify training works end-to-end
**Duration:** 30 minutes

### Quick Test Run

Create `test_training.py`:

```python
# test_training.py
"""
Quick test of training loop on tiny model.
"""

import subprocess
import sys
import os

# Create a tiny config for fast testing
test_config = """
# test_config.py
out_dir = 'out-test'
eval_interval = 10
log_interval = 1
eval_iters = 10
always_save_checkpoint = False
dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 4
block_size = 64

# Tiny model
n_layer = 2
n_head = 2
n_embd = 128
dropout = 0.0

# Short training
learning_rate = 1e-3
max_iters = 50
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR decay
decay_lr = False

# System
device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
dtype = 'float32'
compile = False  # Disable for testing
"""

# Write test config
with open('config/test_config.py', 'w') as f:
    f.write(test_config)

print("Running quick training test...")
print("This will train a tiny model for 50 iterations (~30 seconds)")
print()

# Run training
result = subprocess.run(
    [sys.executable, 'train.py', 'config/test_config.py'],
    capture_output=False
)

if result.returncode == 0:
    print("\n" + "="*60)
    print("✓ Training test successful!")
    print("="*60)

    # Check that checkpoint was created
    if os.path.exists('out-test/ckpt.pt'):
        print("✓ Checkpoint saved successfully")
    else:
        print("✗ Warning: No checkpoint found")
else:
    print("\n" + "="*60)
    print("✗ Training test failed!")
    print("="*60)
    sys.exit(1)
```

### Run Test

```bash
python test_training.py
```

### Expected Output

```
Loading dataset: shakespeare
Train tokens: 304,222
Val tokens: 33,803
Initializing a new model from scratch
Number of parameters: 0.92M
Tokens per iteration: 256
Training for 50 iterations
Effective batch size: 256 tokens
step 0: train loss 10.9876, val loss 10.9912
iter 1: loss 10.9654, time 45.32ms, mfu -100.00%
iter 2: loss 10.8432, time 43.21ms, mfu -100.00%
iter 3: loss 10.6543, time 42.87ms, mfu -100.00%
...
iter 48: loss 4.2345, time 42.56ms, mfu 12.34%
iter 49: loss 4.1987, time 42.43ms, mfu 12.45%
iter 50: loss 4.1654, time 42.39ms, mfu 12.52%
step 50: train loss 4.2134, val loss 4.8765

============================================================
✓ Training test successful!
============================================================
✓ Checkpoint saved successfully
```

### Verification
- [ ] Training runs without errors
- [ ] Loss decreases over time
- [ ] Checkpoint saved
- [ ] Logs show iteration progress
- [ ] MFU is calculated

---

## Running Real Training

### Shakespeare Character-Level Training

```bash
# Prepare data (if not already done)
python data/shakespeare/prepare.py

# Train small model (~5-10 minutes on GPU)
python train.py config/train_shakespeare_char.py
```

Expected output:
```
Loading dataset: shakespeare
Train tokens: 304,222
Val tokens: 33,803
Number of parameters: 10.65M
Tokens per iteration: 16,384
Training for 5000 iterations

step 0: train loss 4.2876, val loss 4.2912
iter 1: loss 4.2654, time 125.32ms, mfu -100.00%
iter 10: loss 3.8432, time 123.21ms, mfu 18.23%
...
step 250: train loss 1.4567, val loss 1.6234
saving checkpoint to out-shakespeare-char
...
step 5000: train loss 0.8123, val loss 1.2456
saving checkpoint to out-shakespeare-char
```

### Fine-tuning GPT-2 on Shakespeare

```bash
# Fine-tune GPT-2 (1.5B) on Shakespeare (~5 minutes)
python train.py config/finetune_shakespeare.py
```

Expected output:
```
Loading dataset: shakespeare
Initializing from OpenAI GPT-2 weights: gpt2-xl
Loading weights from pretrained gpt: gpt2-xl
Number of parameters: 1558.51M
Compiling the model... (takes ~1 minute)

step 0: train loss 3.1234, val loss 3.1456
iter 1: loss 3.0987, time 2345.67ms, mfu 8.45%
...
step 20: train loss 0.9876, val loss 1.2345
saving checkpoint to out-shakespeare-finetuned
```

### Multi-GPU Training (DDP)

```bash
# Train on 4 GPUs
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py
```

Expected output:
```
Loading dataset: openwebtext
Train tokens: 9,035,582,198
Val tokens: 4,434,897
Number of parameters: 124.44M
Tokens per iteration: 196,608 (4 GPUs × 12 batch × 4 accum × 1024 tokens)
Training for 600000 iterations

step 0: train loss 10.9876, val loss 10.9912
iter 1: loss 10.9654, time 187.32ms, mfu -100.00%
iter 100: loss 8.4321, time 135.21ms, mfu 19.23%
...
```

### Verification
- [ ] Shakespeare training completes
- [ ] Loss decreases steadily
- [ ] Checkpoints saved periodically
- [ ] Multi-GPU training works (if testing DDP)

---

## Monitoring Training

### Key Metrics to Watch

1. **Training Loss**:
   - Should decrease steadily
   - Shakespeare: Expect ~0.8-1.2 final loss
   - GPT-2: Expect ~2.8-3.0 final validation loss

2. **Validation Loss**:
   - Should track training loss
   - Gap indicates overfitting (train << val)
   - Shakespeare: Small dataset, some overfitting expected

3. **MFU (Model FLOPs Utilization)**:
   - 10-15%: Typical without optimizations
   - 15-25%: Good (with torch.compile)
   - >25%: Excellent (with flash attention)

4. **Time per Iteration**:
   - Should be consistent after warmup
   - First few iterations slower (compilation)
   - Watch for slowdowns (data loading issues)

### Using TensorBoard (Optional)

Add to `train.py` for TensorBoard logging:

```python
# At top of file
from torch.utils.tensorboard import SummaryWriter

# After master_process check
if master_process:
    writer = SummaryWriter(log_dir=os.path.join(out_dir, 'logs'))

# In training loop
if master_process:
    writer.add_scalar('train/loss', losses['train'], iter_num)
    writer.add_scalar('val/loss', losses['val'], iter_num)
    writer.add_scalar('lr', lr, iter_num)
    writer.add_scalar('mfu', running_mfu, iter_num)
```

View logs:
```bash
tensorboard --logdir=out-shakespeare-char/logs
```

---

## Troubleshooting

### Issue: CUDA out of memory

**Solutions**:
1. Reduce `batch_size`
2. Reduce `block_size` (context length)
3. Increase `gradient_accumulation_steps` (keeps effective batch size same)
4. Use gradient checkpointing (in Phase 6)

### Issue: Loss is NaN

**Causes**:
- Learning rate too high
- Mixed precision issues
- Gradient explosion

**Solutions**:
1. Reduce `learning_rate` (try 1e-4 instead of 6e-4)
2. Ensure `grad_clip` is enabled (set to 1.0)
3. Use `bfloat16` instead of `float16` if available
4. Check data for invalid values

### Issue: Training is very slow

**Solutions**:
1. Enable `compile = True` (2x speedup)
2. Use `bfloat16` or `float16` dtype
3. Increase `batch_size` if memory allows
4. Check GPU utilization: `nvidia-smi` (should be ~90-100%)
5. Profile with `torch.profiler` to find bottlenecks

### Issue: Validation loss not improving

**Possible causes**:
- Model too small for dataset
- Learning rate too high or too low
- Not enough training iterations
- Data quality issues

**Solutions**:
1. Train longer (`max_iters`)
2. Adjust learning rate
3. Increase model size
4. Check data preparation

### Issue: DDP hangs or crashes

**Solutions**:
1. Ensure all GPUs are visible: `echo $CUDA_VISIBLE_DEVICES`
2. Check backend: use `nccl` for GPUs, `gloo` for CPU
3. Verify network setup (for multi-node)
4. Check that `gradient_accumulation_steps` is divisible by world_size

---

## Phase 4 Complete!

### What You've Built

✅ **Complete Training System**:
- Full training loop with optimization
- AdamW optimizer with proper weight decay
- Learning rate scheduling (warmup + cosine decay)
- Mixed precision training (FP16/BF16)
- Gradient accumulation for large effective batches
- Gradient clipping for stability
- Checkpoint saving and resuming
- Evaluation loop with validation loss
- Multi-GPU support with DDP
- Performance metrics (loss, MFU, time)

### Key Files Created

- `train.py` (~300 lines): Complete training script
- `configurator.py`: Config loading utility
- `test_training.py`: Training tests

### Training Statistics

**Shakespeare (Tiny Model)**:
- Model: 6 layers, 384 dim, ~10M params
- Dataset: 300K tokens
- Training: ~5-10 minutes on single GPU
- Final loss: ~0.8-1.2
- Can generate coherent Shakespeare-style text

**GPT-2 Small (124M)**:
- Model: 12 layers, 768 dim, 124M params
- Dataset: 9B tokens (OpenWebText)
- Training: ~3-4 days on 8× A100 GPUs
- Final val loss: ~2.85
- Matches GPT-2 paper performance

### Next Steps

**Proceed to Phase 5: Inference and Generation**
- Implement text generation script
- Autoregressive sampling
- Temperature and top-k sampling
- Interactive generation interface
- Model loading for inference

**Estimated time for Phase 5**: 2-3 hours

---

**Phase 4 Character Count**: ~44,900 characters
