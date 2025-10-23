# Phase 6B: Memory Optimization

## Substage Overview

- **Parent Phase:** Phase 6: Advanced Optimizations
- **Substage:** B of 3 (Memory)
- **Goal:** Implement gradient checkpointing to train larger models with less memory
- **Prerequisites:** Phase 6A completed (speed optimizations)
- **Estimated Duration:** 1.5 hours
- **Key Deliverables:**
  - Gradient checkpointing implementation
  - Enable/disable methods in GPT model
  - Memory profiling tools
  - Memory benchmarking scripts
  - Trade-off analysis (compute vs memory)
  - 30-50% memory reduction

---

## What We're Building

In 6A, we made training faster. Now we make it more memory-efficient:

1. **Gradient Checkpointing** - Trade compute for memory
2. **Memory Profiling** - Understand memory usage
3. **Benchmarking** - Quantify memory savings

**Goal:** Train larger models on the same GPU!

---

## Previous Substage Recap

**Phase 6A - Speed:**
- ✅ torch.compile (2-4x speedup)
- ✅ Flash Attention (2-3x speed + memory)
- ✅ Combined: ~5.5x total speedup

Now we add memory optimization!

---

## Memory Optimization Impact

```
Memory Usage:

Baseline (no optimization)            100%
  ↓
+ Flash Attention                     75%  (from Phase 6A)
  ↓
+ Gradient Checkpointing              40%  ← This substage
  ↓
Combined (Flash + Checkpointing)      30%  (can train 3x larger models!)
```

**Trade-off:** Gradient checkpointing adds ~30% compute time

---

## Step 1: Understanding Gradient Checkpointing

**Duration:** 15 minutes (reading + understanding)

### The Problem

During backpropagation, PyTorch stores all activations from the forward pass:

```
Forward Pass (storing activations):
Input → Layer 1 [STORE] → Layer 2 [STORE] → ... → Layer 12 [STORE] → Output

Memory used = activations for all 12 layers
```

For deep networks, this uses a LOT of memory!

### The Solution

**Gradient checkpointing:**
- DON'T store all activations during forward
- RECOMPUTE them during backward as needed

```
Forward Pass (minimal storage):
Input → Layer 1 → Layer 2 → ... → Layer 12 → Output
         ↑ Only checkpoint a few layers

Backward Pass (recompute as needed):
When we need Layer 6's activations:
  → Recompute forward from last checkpoint
  → Use for gradient computation
  → Discard
```

### Trade-offs

**Benefits:**
- ✅ 30-50% memory reduction
- ✅ Can train larger models
- ✅ Can use larger batch sizes

**Costs:**
- ❌ ~30% slower (recomputation overhead)
- ❌ Not needed if you have enough memory

**When to use:**
- GPU memory is limited
- Want to train larger models
- Can afford slower training
- Batch size limited by memory

---

## Step 2: Implement Gradient Checkpointing

**Purpose:** Add checkpointing to GPT model  
**Duration:** 20 minutes

### Add to model.py

Add these methods to the `GPT` class in `model.py`:

```python
# In GPT.__init__, add:
self.gradient_checkpointing = False  # Disabled by default

# Add these methods to GPT class:

def enable_gradient_checkpointing(self):
    """Enable gradient checkpointing to save memory."""
    self.gradient_checkpointing = True
    print("Gradient checkpointing enabled")

def disable_gradient_checkpointing(self):
    """Disable gradient checkpointing."""
    self.gradient_checkpointing = False
```

### Update GPT.forward()

Modify the transformer block loop in `forward()`:

```python
# In GPT.forward(), replace:
for block in self.transformer.h:
    x = block(x)

# With:
for block in self.transformer.h:
    if self.gradient_checkpointing and self.training:
        # Use gradient checkpointing
        x = torch.utils.checkpoint.checkpoint(
            block,
            x,
            use_reentrant=False  # Recommended for PyTorch 2.0+
        )
    else:
        # Normal forward pass
        x = block(x)
```

### Add Config Option to train.py

Add to config section in `train.py`:

```python
# Gradient checkpointing (Phase 6B)
gradient_checkpointing = False  # Trade compute for memory
```

Add after model creation:

```python
# Enable gradient checkpointing if requested
if gradient_checkpointing:
    raw_model = model.module if ddp else model
    raw_model.enable_gradient_checkpointing()
    print("Gradient checkpointing enabled (trades compute for memory)")
```

---

## Step 3: Memory Benchmarking

**Purpose:** Measure memory savings  
**Duration:** 30 minutes

### Create benchmark_memory.py

```python
# benchmark_memory.py
"""
Benchmark memory usage with/without gradient checkpointing.
"""

import torch
from model import GPT, GPTConfig


def benchmark_memory(use_checkpointing=False):
    """
    Benchmark memory usage.

    Args:
        use_checkpointing: Enable gradient checkpointing

    Returns:
        float: Peak memory usage in GB
    """
    # Create model
    config = GPTConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=1024,
        vocab_size=50304,
        dropout=0.0
    )

    device = 'cuda'
    model = GPT(config).to(device)

    if use_checkpointing:
        model.enable_gradient_checkpointing()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Forward + backward
    batch_size = 8
    seq_len = 1024
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training step
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(x, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Get peak memory
    peak_memory = torch.cuda.max_memory_allocated() / 1e9

    return peak_memory


if __name__ == '__main__':
    print("=" * 80)
    print("Gradient Checkpointing Memory Benchmark")
    print("=" * 80 + "\n")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        exit(0)

    # Without checkpointing
    print("1. Without gradient checkpointing:")
    mem_without = benchmark_memory(use_checkpointing=False)
    print(f"   Peak memory: {mem_without:.2f} GB\n")

    # With checkpointing
    print("2. With gradient checkpointing:")
    mem_with = benchmark_memory(use_checkpointing=True)
    print(f"   Peak memory: {mem_with:.2f} GB\n")

    # Results
    reduction = (1 - mem_with / mem_without) * 100
    print("=" * 80)
    print(f"Memory reduction: {reduction:.1f}%")
    print("=" * 80)

    if reduction > 30:
        print("✓ Significant memory savings!")
    else:
        print("⚠ Memory savings lower than expected")
```

### Run Benchmark

```bash
python benchmark_memory.py
```

### Expected Output

```
================================================================================
Gradient Checkpointing Memory Benchmark
================================================================================

1. Without gradient checkpointing:
   Peak memory: 8.45 GB

2. With gradient checkpointing:
   Peak memory: 5.23 GB

================================================================================
Memory reduction: 38.1%
================================================================================
✓ Significant memory savings!
```

---

## Step 4: When to Use Gradient Checkpointing

**Purpose:** Understand the trade-offs  
**Duration:** 10 minutes

### Decision Matrix

**Use gradient checkpointing if:**
- ✅ GPU memory is limited
- ✅ Want to train larger models
- ✅ Want to use larger batch sizes
- ✅ Can afford ~30% slower training

**Don't use if:**
- ❌ Have plenty of GPU memory
- ❌ Need maximum training speed
- ❌ Model already fits comfortably

### Example Scenarios

**Scenario 1: Limited Memory**
```python
# Without checkpointing: OOM (Out of Memory)
batch_size = 16
gradient_checkpointing = False  # Crashes!

# With checkpointing: Works!
batch_size = 16
gradient_checkpointing = True  # Success!
```

**Scenario 2: Larger Models**
```python
# Without checkpointing: Can only do 12 layers
n_layer = 12
gradient_checkpointing = False

# With checkpointing: Can do 24 layers!
n_layer = 24
gradient_checkpointing = True
```

---

## Step 5: Memory Profiling

**Purpose:** Understand where memory is used  
**Duration:** 15 minutes

### Create profile_memory.py

```python
# profile_memory.py
"""
Profile memory usage during training.
"""

import torch
from model import GPT, GPTConfig


def profile_memory():
    """Profile memory usage."""
    print("=" * 80)
    print("Memory Profiling")
    print("=" * 80 + "\n")

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    config = GPTConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=1024,
        vocab_size=50304,
        dropout=0.0
    )

    device = 'cuda'
    
    # Test with checkpointing OFF
    print("Testing WITHOUT gradient checkpointing:\n")
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    model = GPT(config).to(device)
    
    print(f"After model creation: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Create data
    x = torch.randint(0, config.vocab_size, (8, 1024), device=device)
    y = torch.randint(0, config.vocab_size, (8, 1024), device=device)
    
    # Forward
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(x, y)
    
    print(f"After forward pass: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Backward
    loss.backward()
    
    print(f"After backward pass: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    
    # Test with checkpointing ON
    print("\n" + "-" * 80)
    print("Testing WITH gradient checkpointing:\n")
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    model = GPT(config).to(device)
    model.enable_gradient_checkpointing()
    
    print(f"After model creation: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Forward
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(x, y)
    
    print(f"After forward pass: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Backward
    loss.backward()
    
    print(f"After backward pass: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == '__main__':
    profile_memory()
```

### Run Profiling

```bash
python profile_memory.py
```

### Expected Output

```
================================================================================
Memory Profiling
================================================================================

Testing WITHOUT gradient checkpointing:

After model creation: 0.95 GB
After forward pass: 6.23 GB
After backward pass: 8.45 GB
Peak memory: 8.45 GB

--------------------------------------------------------------------------------
Testing WITH gradient checkpointing:

After model creation: 0.95 GB
After forward pass: 3.12 GB
After backward pass: 5.23 GB
Peak memory: 5.23 GB
```

**Analysis:**
- Model weights: ~1 GB (same in both cases)
- Activations: Reduced from 5.3 GB → 2.2 GB (~60% reduction!)
- Total memory: 8.45 GB → 5.23 GB (38% reduction)

---

## Troubleshooting

### Issue: "checkpoint expects a function"

**Problem:** Incorrect usage of torch.utils.checkpoint

**Solution:**
- Must pass a Module (like `block`), not a method
- Correct: `checkpoint(block, x)`
- Incorrect: `checkpoint(block.forward, x)`

### Issue: Still running out of memory

**Solutions:**
1. Reduce batch_size further
2. Reduce block_size (context length)
3. Use smaller model
4. Combine with Flash Attention (from 6A)

### Issue: Training much slower

**Expected!**
- Gradient checkpointing adds ~30% overhead
- This is the trade-off for memory savings
- If speed is critical, don't use checkpointing

---

## Phase 6B Complete!

### What You've Built

✅ **Memory Optimizations**:
- Gradient checkpointing implementation
- Enable/disable methods in GPT class
- Memory benchmarking scripts
- Memory profiling tools
- Trade-off analysis

### Key Files Created/Modified

- `model.py`: Added gradient checkpointing support
- `train.py`: Added gradient_checkpointing config option
- `benchmark_memory.py`: Memory benchmarks
- `profile_memory.py`: Memory profiling

### Memory Improvements

**Typical savings:**
- Gradient checkpointing: 30-50% memory reduction
- Combined with Flash Attention: 50-70% total reduction
- Can train 2-3x larger models on same GPU

**Performance impact:**
- ~30% slower training (recomputation overhead)
- Worth it when memory-constrained

### Example Use Case

**Before:**
```python
# GPU with 16GB memory
n_layer = 12  # GPT-2 Small
batch_size = 8
# Uses ~15GB (barely fits)
```

**After (with checkpointing):**
```python
# Same GPU with 16GB memory
n_layer = 24  # GPT-2 Medium!
batch_size = 8
# Uses ~10GB (comfortable headroom)
```

---

## Next Substage

**Proceed to Phase 6C: Distributed Training & Profiling**

Final optimization substage covers:
- Multi-node distributed training (scale across machines)
- SLURM integration (for HPC clusters)
- Performance profiling with PyTorch profiler
- Comprehensive benchmarking
- Bottleneck identification

**Estimated time:** 2 hours

This is advanced infrastructure for serious training!

---

**Phase 6B Character Count:** ~9,800 characters
