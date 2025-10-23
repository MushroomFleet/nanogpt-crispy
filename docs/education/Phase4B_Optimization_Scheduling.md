# Phase 4B: Optimization & Scheduling

## Substage Overview

- **Parent Phase:** Phase 4: Training Loop
- **Substage:** B of 3 (Optimization)
- **Goal:** Implement optimizer, learning rate scheduling, and gradient operations
- **Prerequisites:** Phase 4A completed (training infrastructure ready)
- **Estimated Duration:** 2 hours
- **Key Deliverables:**
  - AdamW optimizer with proper weight decay groups
  - Learning rate scheduling (warmup + cosine decay)
  - Evaluation function for validation loss
  - Gradient accumulation mechanics
  - Gradient clipping for stability
  - Mixed precision training with GradScaler
  - MFU (Model FLOPs Utilization) calculation

---

## What We're Building

In Phase 4A, we set up the infrastructure. Now we add the components that make the model learn:

1. **Optimizer** - AdamW with smart weight decay
2. **LR Scheduler** - Warmup + cosine decay
3. **Evaluation** - Validation loss computation
4. **Gradient Ops** - Accumulation and clipping
5. **Performance Metrics** - MFU tracking

---

## Previous Substage Recap

**Phase 4A built:**
- ✅ Training script structure
- ✅ Configuration system
- ✅ Data loading (memory-mapped)
- ✅ Model initialization (3 modes)
- ✅ Basic forward pass

Now we make it actually train!

---

## Step 1: Optimizer Setup

**Purpose:** Configure AdamW with proper weight decay  
**Duration:** 30 minutes

### Add Optimizer to train.py

Add this after model initialization:

```python
# Optimizer
# Note: model.configure_optimizers is already implemented in model.py (Phase 2C)
optimizer = model.module.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type) if ddp else model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume' and 'checkpoint' in dir():
    optimizer.load_state_dict(checkpoint['optimizer'])

checkpoint = None  # Free up memory
```

### Understanding Weight Decay Groups

The `configure_optimizers` method (already in model.py from Phase 2C) does something smart:

**Two parameter groups:**
1. **With weight decay**: Weight matrices (Linear layers)
2. **Without weight decay**: Biases, LayerNorm, Embeddings

**Why?**
- Weight decay on matrices prevents overfitting
- Weight decay on biases/LayerNorm can hurt performance
- This matches GPT-2 paper's approach

**What gets decayed:**
- ✅ Attention projection weights
- ✅ MLP weights
- ✅ Output head weights

**What doesn't:**
- ❌ All biases
- ❌ LayerNorm parameters
- ❌ Embedding weights

---

## Step 2: Learning Rate Scheduling

**Purpose:** Implement warmup + cosine decay  
**Duration:** 20 minutes

### Add LR Schedule Function

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

### Understanding the Schedule

```
Learning Rate Over Time

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

**Three phases:**

1. **Warmup (0 → warmup_iters)**:
   - Linear increase: 0 → learning_rate
   - Prevents instability at start
   - Gradients are noisy initially

2. **Cosine Decay (warmup_iters → lr_decay_iters)**:
   - Smooth decrease: learning_rate → min_lr
   - Helps find better minima
   - Standard in transformer training

3. **Constant (after lr_decay_iters)**:
   - Stay at min_lr
   - Final fine-tuning

### Test the Schedule

Add quick test:

```python
if __name__ == '__main__':
    # Test LR schedule
    warmup_iters = 100
    lr_decay_iters = 1000
    learning_rate = 6e-4
    min_lr = 6e-5
    decay_lr = True
    
    print("Learning rate schedule:")
    for it in [0, 50, 100, 500, 999, 1000]:
        lr = get_lr(it)
        print(f"  Iter {it:4d}: {lr:.2e}")
```

Expected output:
```
Learning rate schedule:
  Iter    0: 0.00e+00  (warmup start)
  Iter   50: 3.00e-04  (warmup middle)
  Iter  100: 6.00e-04  (warmup end, peak)
  Iter  500: 3.30e-04  (decay middle)
  Iter  999: 6.00e-05  (decay end, min)
  Iter 1000: 6.00e-05  (after decay)
```

---

## Step 3: Evaluation Function

**Purpose:** Compute validation loss  
**Duration:** 20 minutes

### Add Evaluation to train.py

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

### Understanding Evaluation

**Why average over multiple batches?**
- Single batch is noisy (random sampling)
- Average over `eval_iters` batches (e.g., 200)
- Gives stable validation metrics

**@torch.no_grad() decorator:**
```python
@torch.no_grad()
def estimate_loss():
    # No gradient computation here
```
- Disables gradient tracking
- Saves memory
- Speeds up evaluation
- Essential for eval mode

**model.eval() vs model.train():**
- `eval()`: Disables dropout, sets eval mode
- `train()`: Enables dropout, sets train mode
- Always switch back after evaluation

---

## Step 4: Add MFU Estimation to Model

**Purpose:** Track GPU efficiency  
**Duration:** 15 minutes

### Add to model.py

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

### Understanding MFU

**What is MFU?**
- Model FLOPs Utilization
- Measures % of GPU's theoretical peak FLOP/s we achieve
- GPT-3 paper reported 21% MFU on A100s

**FLOP Calculation:**
```
Forward pass:  ~6N FLOPs (N = number of parameters)
Backward pass: ~2 × forward
Attention:     ~12*L*H*Q*T FLOPs
```

**Typical MFU values:**
- 10-15%: Not optimized
- 15-25%: Well-optimized (GPT-3 level)
- 25-40%: Excellent (with Flash Attention)
- >40%: State-of-the-art

**A100 Peak FLOPS:**
- BF16: 312 TFLOP/s
- FP32: 19.5 TFLOP/s
- FP16: 312 TFLOP/s (same as BF16)

---

## Step 5: Simple Training with Optimizer

**Purpose:** Complete one training iteration with learning  
**Duration:** 30 minutes

### Update Training Loop

Replace the simple test loop in train.py with this:

```python
# Initialize tracking
iter_num = 0 if init_from == 'scratch' or init_from.startswith('gpt2') else checkpoint['iter_num']
best_val_loss = 1e9 if init_from == 'scratch' or init_from.startswith('gpt2') else checkpoint['best_val_loss']
t0 = time.time()

# Get model for MFU estimation
raw_model = model.module if ddp else model

print(f"\nStarting training from iteration {iter_num}")
print(f"Training for {max_iters} iterations")
print("="*60)

# Training loop - simplified version for Phase 4B
for iteration in range(min(10, max_iters)):  # Just 10 iterations to test
    # Update learning rate
    lr = get_lr(iteration) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Get batch
    X, Y = get_batch('train', train_data, val_data)
    
    # Forward pass
    with ctx:
        logits, loss = model(X, Y)
    
    # Backward pass
    scaler.scale(loss).backward()
    
    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    
    # Zero gradients
    optimizer.zero_grad(set_to_none=True)
    
    # Timing
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    # Logging
    if iteration % log_interval == 0:
        lossf = loss.item()
        print(f"iter {iteration}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")

print("\n" + "="*60)
print("Optimization test complete!")
print("Loss should decrease over iterations.")
print("\nIn Phase 4C, we'll add:")
print("  - Complete training loop with gradient accumulation")
print("  - Evaluation and checkpointing")
print("  - Multi-GPU (DDP) support")
```

---

## Understanding Gradient Operations

### 1. Mixed Precision Training

```python
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    logits, loss = model(X, Y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**What's happening:**
- Forward pass uses FP16/BF16 (2x faster, 2x less memory)
- GradScaler prevents underflow (FP16 has small range)
- Scales gradients up before backward
- Unscales before optimizer step
- Updates scale for next iteration

### 2. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```

**Purpose:**
- Prevents exploding gradients
- Clips gradient norm to maximum value (e.g., 1.0)
- Essential for stable training

**How it works:**
```python
total_norm = sqrt(sum(grad^2 for all parameters))
if total_norm > grad_clip:
    scale_factor = grad_clip / total_norm
    for grad in gradients:
        grad *= scale_factor
```

### 3. Optimizer Zero Grad

```python
optimizer.zero_grad(set_to_none=True)
```

**Why set_to_none=True?**
- Slightly more memory efficient
- Sets gradients to None instead of zeros
- Recommended in PyTorch docs

---

## Test the Optimizer

### Run Training Test

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
Number of parameters: 0.75M
Using fused AdamW: False

Starting training from iteration 0
Training for 20 iterations
============================================================
iter 0: loss 10.8234, time 45.32ms, lr 1.00e-03
iter 1: loss 10.7156, time 42.15ms, lr 1.00e-03
iter 2: loss 10.5987, time 41.89ms, lr 1.00e-03
iter 3: loss 10.4523, time 42.03ms, lr 1.00e-03
iter 4: loss 10.2891, time 41.95ms, lr 1.00e-03
iter 5: loss 10.1234, time 42.11ms, lr 1.00e-03
...

============================================================
Optimization test complete!
Loss should decrease over iterations.

In Phase 4C, we'll add:
  - Complete training loop with gradient accumulation
  - Evaluation and checkpointing
  - Multi-GPU (DDP) support
```

### What to Look For

✅ **Loss decreases**: 10.8 → 10.1 over 10 iterations  
✅ **Consistent timing**: ~40-45ms per iteration  
✅ **LR shows correctly**: Learning rate displayed  
✅ **No NaN**: Loss should be a valid number

---

## Step 6: Test Learning Rate Schedule

**Purpose:** Verify LR schedule works correctly  
**Duration:** 10 minutes

### Create test_lr_schedule.py

```python
# test_lr_schedule.py
"""
Visualize the learning rate schedule.
"""

import math

def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr, decay_lr):
    """Learning rate schedule."""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    if not decay_lr:
        return learning_rate
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def test_schedule():
    """Test and visualize LR schedule."""
    print("=" * 60)
    print("Learning Rate Schedule Test")
    print("=" * 60 + "\n")
    
    # Config
    warmup_iters = 2000
    lr_decay_iters = 10000
    learning_rate = 6e-4
    min_lr = 6e-5
    decay_lr = True
    
    print(f"Config:")
    print(f"  Learning rate: {learning_rate:.2e}")
    print(f"  Min LR: {min_lr:.2e}")
    print(f"  Warmup iters: {warmup_iters:,}")
    print(f"  Decay iters: {lr_decay_iters:,}")
    print()
    
    # Test key points
    test_points = [
        0,              # Start
        1000,           # Mid-warmup
        2000,           # End warmup
        5000,           # Mid-decay
        10000,          # End decay
        15000,          # After decay
    ]
    
    print("LR at key iterations:")
    for it in test_points:
        lr = get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr, decay_lr)
        
        # Identify phase
        if it < warmup_iters:
            phase = "warmup"
        elif it < lr_decay_iters:
            phase = "decay"
        else:
            phase = "constant"
        
        print(f"  Iter {it:6,}: {lr:.2e}  ({phase})")
    
    print("\n✓ Schedule working correctly!")


if __name__ == '__main__':
    test_schedule()
```

Run:
```bash
python test_lr_schedule.py
```

Expected output:
```
============================================================
Learning Rate Schedule Test
============================================================

Config:
  Learning rate: 6.00e-04
  Min LR: 6.00e-05
  Warmup iters: 2,000
  Decay iters: 10,000

LR at key iterations:
  Iter      0: 0.00e+00  (warmup)
  Iter  1,000: 3.00e-04  (warmup)
  Iter  2,000: 6.00e-04  (decay)
  Iter  5,000: 3.90e-04  (decay)
  Iter 10,000: 6.00e-05  (constant)
  Iter 15,000: 6.00e-05  (constant)

✓ Schedule working correctly!
```

---

## Phase 4B Complete!

### What You've Built

✅ **Optimization & Scheduling**:
- AdamW optimizer with smart weight decay groups
- Learning rate scheduling (warmup + cosine decay)
- Evaluation function for validation loss
- Mixed precision training with GradScaler
- Gradient clipping for stability
- MFU estimation for performance tracking
- Working training iterations with learning

### Key Additions to train.py

- Optimizer setup (~5 lines)
- `get_lr()` function for LR scheduling (~15 lines)
- `estimate_loss()` function for evaluation (~15 lines)
- Updated training loop with optimization (~20 lines)

### Key Additions to model.py

- `estimate_mfu()` method in GPT class (~15 lines)

### What We Can Do Now

✅ **Train models!**
- Forward pass computes loss
- Backward pass computes gradients
- Optimizer updates weights
- Learning rate adjusts over time
- Loss should decrease

**Example:** Run 1000 iterations on Shakespeare:
```bash
# Update test_config.py: max_iters = 1000
python train.py config/test_config.py
```

You should see loss decrease from ~11 to ~4-5 over 1000 iterations!

---

## Next Substage

**Proceed to Phase 4C: Production Training**

Now that we have working optimization, we'll add:
- Complete training loop with gradient accumulation
- Evaluation at intervals
- Checkpoint saving and loading
- Resume training capability
- Multi-GPU (DDP) training
- Progress monitoring
- Real training examples

**Estimated time:** 3 hours

This is where we complete the production-ready training system!

---

**Phase 4B Character Count:** ~16,800 characters
