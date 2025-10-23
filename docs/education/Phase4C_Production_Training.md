# Phase 4C: Production Training

## Substage Overview

- **Parent Phase:** Phase 4: Training Loop
- **Substage:** C of 3 (Production)
- **Goal:** Complete the production-ready training system
- **Prerequisites:** Phase 4A & 4B completed (infrastructure + optimization ready)
- **Estimated Duration:** 3 hours
- **Key Deliverables:**
  - Complete training loop with gradient accumulation
  - Periodic evaluation
  - Checkpoint saving and loading
  - Resume training capability
  - Multi-GPU (DDP) training support
  - Progress monitoring and logging
  - Real training examples (Shakespeare, GPT-2 fine-tuning)
  - Comprehensive troubleshooting guide

---

## What We're Building

We have infrastructure (4A) and optimization (4B). Now we complete the system:

1. **Full Training Loop** - Gradient accumulation, evaluation, checkpointing
2. **DDP Support** - Multi-GPU training
3. **Monitoring** - Track progress, save checkpoints
4. **Real Training** - Actually train models to completion

This is the final piece that makes everything production-ready!

---

## Previous Substages Recap

**Phase 4A - Infrastructure:**
- ✅ Training script structure
- ✅ Configuration system
- ✅ Data loading
- ✅ Model initialization

**Phase 4B - Optimization:**
- ✅ AdamW optimizer
- ✅ Learning rate scheduling
- ✅ Evaluation function
- ✅ Gradient operations

Now we put it all together!

---

## Step 1: Complete Training Loop

**Purpose:** Implement full training with all features  
**Duration:** 60-90 minutes

### Replace Training Loop in train.py

Replace the simple 10-iteration loop with this complete version:

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

---

## Step 2: Understanding Key Features

### Gradient Accumulation

```python
for micro_step in range(gradient_accumulation_steps):
    loss = loss / gradient_accumulation_steps
    loss.backward()
optimizer.step()
```

**Purpose:**
- Simulate larger batch sizes
- Useful when GPU memory is limited

**Example:**
- `gradient_accumulation_steps=4`, `batch_size=12`
- Effective batch size = 4 × 12 = 48
- Same gradients as batch_size=48, but uses less memory

**How it works:**
1. Forward/backward 4 times (accumulate gradients)
2. One optimizer step (update weights)
3. Zero gradients
4. Repeat

### DDP Gradient Sync

```python
if ddp:
    model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
```

**Purpose:**
- Only sync gradients on last micro-step
- Saves communication bandwidth
- Faster multi-GPU training

### Checkpointing

```python
checkpoint = {
    'model': raw_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
    'best_val_loss': best_val_loss,
    'config': config,
}
torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
```

**What's saved:**
- Model weights
- Optimizer state (momentum, etc.)
- Training iteration
- Best validation loss
- Full configuration

**When saved:**
- When validation loss improves
- Or every eval if `always_save_checkpoint=True`

---

## Step 3: Real Training Example

**Purpose:** Train a small Shakespeare model  
**Duration:** 10 minutes setup + 5-10 min training

### Update config/train_shakespeare_char.py

Ensure you have this config (should exist from Phase 1):

```python
# config/train_shakespeare_char.py
"""
Train a character-level GPT on Shakespeare text.
Small model for fast training and experimentation.
"""

# I/O
out_dir = 'out-shakespeare-char'
eval_interval = 250
log_interval = 10
eval_iters = 200
always_save_checkpoint = True

# Data
dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

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
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4

# System
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # Set to True if PyTorch 2.0+
```

### Run Training

```bash
# Ensure data is prepared
python data/shakespeare/prepare.py

# Train the model (5-10 minutes on GPU)
python train.py config/train_shakespeare_char.py
```

### Expected Output

```
Overriding config with config/train_shakespeare_char.py:
Tokens per iteration: 16,384
Loading dataset: shakespeare
Train tokens: 304,222
Val tokens: 33,803
Initializing model from 'scratch'
Number of parameters: 10.65M
Using fused AdamW: False

Training for 5000 iterations
Effective batch size: 16,384 tokens
============================================================
step 0: train loss 4.2876, val loss 4.2912
iter 10: loss 4.1234, time 125.32ms, mfu -100.00%
iter 20: loss 3.9876, time 123.45ms, mfu 18.23%
iter 30: loss 3.8543, time 122.87ms, mfu 18.45%
...
step 250: train loss 1.4567, val loss 1.6234
saving checkpoint to out-shakespeare-char
iter 260: loss 1.4321, time 123.11ms, mfu 19.12%
...
step 500: train loss 1.2345, val loss 1.4123
saving checkpoint to out-shakespeare-char
...
step 5000: train loss 0.8123, val loss 1.2456
saving checkpoint to out-shakespeare-char
```

### What to Observe

**During training:**
- ✅ Train loss decreases steadily: 4.3 → 0.8
- ✅ Val loss decreases: 4.3 → 1.2
- ✅ Gap appears (train < val): Some overfitting (normal for small dataset)
- ✅ MFU stabilizes: ~15-20% without optimizations
- ✅ Checkpoints saved every 250 iterations

**Training time:**
- Small GPU: ~10 minutes
- Large GPU: ~5 minutes
- CPU: ~2 hours (not recommended)

---

## Step 4: Testing the Trained Model

**Purpose:** Verify the model learned something  
**Duration:** 5 minutes

### Quick Generation Test

After training completes, test generation:

```bash
python sample.py --out_dir=out-shakespeare-char --start="\n" --num_samples=3
```

You should see Shakespeare-style text! It won't be perfect, but should be coherent.

Example output:
```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?
```

---

## Step 5: Multi-GPU Training (Optional)

**Purpose:** Enable DDP for faster training  
**Duration:** 15 minutes

### Running on Multiple GPUs

```bash
# 4 GPUs on one machine
torchrun --standalone --nproc_per_node=4 train.py config/train_shakespeare_char.py
```

### Expected Behavior

**With 4 GPUs:**
- Effective batch size = 4 × 64 = 256
- Each GPU processes 64 samples
- Gradients synchronized across GPUs
- ~4x speedup (not perfect due to sync overhead)

**Logging:**
```
Rank 0/4 initialized
Rank 1/4 initialized
Rank 2/4 initialized
Rank 3/4 initialized
Tokens per iteration: 65,536  (4x more than single GPU)
...
```

**Notes:**
- Only rank 0 (master process) logs and saves checkpoints
- Other ranks train silently
- All ranks must finish together

---

## Step 6: Resume Training

**Purpose:** Test checkpoint resume functionality  
**Duration:** 10 minutes

### Interrupt and Resume

1. **Start training:**
```bash
python train.py config/train_shakespeare_char.py
```

2. **Interrupt (Ctrl+C) after a few hundred iterations**

3. **Resume:**
```bash
# Update config to resume
# In train_shakespeare_char.py, change:
# init_from = 'resume'

python train.py config/train_shakespeare_char.py
```

### Expected Output

```
Initializing model from 'resume'
Resuming training from out-shakespeare-char
Loading checkpoint...
Resuming from iteration 250
Best val loss so far: 1.6234

step 250: train loss 1.4567, val loss 1.6234
iter 260: loss 1.4321, time 123.11ms, mfu 19.12%
...
```

**Verification:**
- ✅ Continues from saved iteration
- ✅ Validation loss matches
- ✅ Training continues smoothly

---

## Troubleshooting

### Issue: CUDA out of memory

**Solutions:**
1. Reduce `batch_size`
2. Reduce `block_size` (context length)
3. Increase `gradient_accumulation_steps` (same effective batch, less memory)
4. Use smaller model

**Example fix:**
```python
# In config file:
batch_size = 32  # Was 64
gradient_accumulation_steps = 2  # Was 1
# Effective batch stays same: 32*2 = 64
```

### Issue: Loss is NaN

**Causes:**
- Learning rate too high
- Mixed precision underflow
- Bad data

**Solutions:**
1. Reduce `learning_rate` (try 1e-4 instead of 1e-3)
2. Ensure `grad_clip` enabled (set to 1.0)
3. Use `bfloat16` instead of `float16` if available
4. Check data: `assert not torch.isnan(X).any()`

### Issue: Training very slow

**Solutions:**
1. Enable `compile = True` (2x speedup, PyTorch 2.0+)
2. Use `bfloat16` or `float16` dtype
3. Increase `batch_size` if memory allows
4. Check GPU utilization: `nvidia-smi` (should be 90-100%)

### Issue: Validation loss not improving

**Causes:**
- Model too small
- Learning rate too high/low
- Not enough iterations
- Data quality

**Solutions:**
1. Train longer (`max_iters`)
2. Adjust learning rate
3. Increase model size
4. Check data preparation

### Issue: DDP hangs

**Solutions:**
1. Check all GPUs visible: `nvidia-smi`
2. Verify `gradient_accumulation_steps % world_size == 0`
3. Ensure same PyTorch version on all processes
4. Use `backend='nccl'` for GPUs

---

## Monitoring Training Progress

### Key Metrics to Watch

**1. Training Loss:**
- Should decrease steadily
- Shakespeare: ~4.3 → ~0.8 over 5000 iterations
- GPT-2: ~11 → ~3 over 600k iterations

**2. Validation Loss:**
- Should track training loss
- Small gap is normal
- Large gap (train << val) indicates overfitting

**3. MFU:**
- Should stabilize after ~5 iterations
- 10-15%: Typical without optimizations
- 15-25%: Good (with compile)
- >25%: Excellent (with Flash Attention)

**4. Time per Iteration:**
- Should be consistent
- First iteration slower (compilation)
- Slowdowns indicate data loading issues

### Optional: TensorBoard Logging

Add to top of train.py:

```python
# Optional TensorBoard logging
try:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(out_dir, 'logs')) if master_process else None
except ImportError:
    writer = None
```

Add in evaluation section:

```python
if writer is not None:
    writer.add_scalar('train/loss', losses['train'], iter_num)
    writer.add_scalar('val/loss', losses['val'], iter_num)
    writer.add_scalar('lr', lr, iter_num)
```

View logs:
```bash
tensorboard --logdir=out-shakespeare-char/logs
```

---

## Real Training Examples

### Example 1: Shakespeare Character-Level

**Configuration:** `config/train_shakespeare_char.py`

**Training:**
```bash
python train.py config/train_shakespeare_char.py
```

**Expected Results:**
- Training time: 5-10 minutes (GPU)
- Final train loss: ~0.8-1.0
- Final val loss: ~1.2-1.4
- Model can generate coherent Shakespeare-style text

**Generation:**
```bash
python sample.py --out_dir=out-shakespeare-char --start="ROMEO:" --num_samples=3
```

---

### Example 2: Fine-Tune GPT-2 on Shakespeare

**Configuration:** `config/finetune_shakespeare.py`

Create this config:

```python
# config/finetune_shakespeare.py
"""
Fine-tune GPT-2 XL on Shakespeare.
Very fast (~5 minutes) and produces excellent results.
"""

# I/O
out_dir = 'out-shakespeare-finetuned'
eval_interval = 5
log_interval = 1
eval_iters = 20
always_save_checkpoint = False

# Data
dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 1
block_size = 1024

# Model - Start from GPT-2
init_from = 'gpt2-xl'  # 1.5B parameters

# Fine-tuning hyperparameters
learning_rate = 3e-5  # Much smaller than pre-training
max_iters = 20  # Very few iterations needed
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 0  # No warmup for fine-tuning
lr_decay_iters = 20
min_lr = 3e-5

# System
import torch
device = 'cuda'  # Requires GPU
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
compile = False  # Can't compile when loading from HF
```

**Training:**
```bash
python train.py config/finetune_shakespeare.py
```

**Expected:**
- Downloads GPT-2 XL (~6GB)
- Fine-tunes for 20 iterations (~5 minutes)
- Produces high-quality Shakespeare text

---

## Phase 4C Complete!

### What You've Built

✅ **Complete Training System**:
- Full training loop with gradient accumulation
- Periodic evaluation (train & val loss)
- Checkpoint saving (model + optimizer + state)
- Resume training from checkpoints
- Multi-GPU support (DDP)
- Performance monitoring (MFU, timing)
- Real training examples
- Comprehensive troubleshooting

### Key Files

- `train.py`: Complete training script (~300 lines)
- `configurator.py`: Config loader
- `config/train_shakespeare_char.py`: Shakespeare training
- `config/finetune_shakespeare.py`: GPT-2 fine-tuning
- `config/test_config.py`: Quick testing

### Training Capabilities

**You can now:**
1. ✅ Train models from scratch
2. ✅ Resume interrupted training
3. ✅ Fine-tune GPT-2 on custom data
4. ✅ Train on single or multiple GPUs
5. ✅ Monitor progress with metrics
6. ✅ Save and load checkpoints

### Training Statistics

**Shakespeare (Small Model):**
- Model: 6 layers, 384 dim, ~10M params
- Dataset: 300K tokens
- Training: ~5-10 minutes on single GPU
- Final loss: ~0.8-1.2
- Can generate coherent Shakespeare text

**GPT-2 Fine-Tuning:**
- Model: 48 layers, 1600 dim, ~1.5B params
- Dataset: 300K tokens
- Training: ~5 minutes on high-end GPU
- Final loss: ~1.0
- Produces excellent Shakespeare-style text

---

## Next Phase

**Proceed to Phase 5: Inference and Generation**

Now that you can train models, Phase 5 covers:
- Complete generation script (`sample.py`)
- Advanced sampling strategies
- Interactive generation
- Batch generation
- Quality tuning

**Estimated time:** 2-3 hours

Or skip to **Phase 6: Advanced Optimizations** for:
- torch.compile (2-4x speedup)
- Flash Attention (memory optimization)
- Gradient checkpointing
- Multi-node training
- Performance profiling

---

**Phase 4C Character Count:** ~19,200 characters
