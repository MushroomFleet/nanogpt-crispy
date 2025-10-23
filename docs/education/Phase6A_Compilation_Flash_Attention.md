# Phase 6A: Compilation & Flash Attention

## Substage Overview

- **Parent Phase:** Phase 6: Advanced Optimizations
- **Substage:** A of 3 (Speed Optimizations)
- **Goal:** Implement PyTorch 2.0 compilation and Flash Attention for major speedups
- **Prerequisites:** Phase 1-5 completed (working training pipeline)
- **Estimated Duration:** 1.5 hours
- **Key Deliverables:**
  - PyTorch 2.0 torch.compile integration
  - Advanced compilation options
  - Flash Attention support (optional, requires CUDA)
  - Compilation benchmarking script
  - Attention performance benchmarking
  - 2-4x speed improvement

---

## What We're Building

Phase 6 focuses on optimizations. In this substage, we'll add the two biggest speed wins:

1. **torch.compile** - PyTorch 2.0 feature for 2x+ speedup
2. **Flash Attention** - Memory-efficient attention for 2-3x speed + memory savings

These are independent optimizations that stack multiplicatively!

---

## Optimization Impact

```
Performance Gains:

Baseline (no optimizations)           1.0x
  ↓
+ torch.compile                       2.0x  ← This substage
  ↓
+ Flash Attention                     4.0x  ← This substage
  ↓
Combined (compile + flash)            ~5.5x total
```

**Both are optional but highly recommended!**

---

## Step 1: PyTorch 2.0 Compilation

**Purpose:** Enable torch.compile for 2x+ speedup  
**Duration:** 30 minutes

### Understanding torch.compile

PyTorch 2.0 introduced `torch.compile()`:
- Traces model computation graph
- Optimizes kernels using TorchInductor
- Fuses operations for efficiency
- Generates specialized CUDA code

**Benefits:**
- 2-4x speedup (forward + backward)
- No code changes needed
- Works with existing models

**Limitations:**
- First iteration slow (compilation overhead ~1 min)
- Requires PyTorch 2.0+
- Some dynamic operations not supported

### Already Integrated!

Good news: Basic compilation is already in `train.py` from Phase 4:

```python
# In train.py, after model creation
if compile:
    print("Compiling the model... (takes ~1 minute)")
    unoptimized_model = model
    model = torch.compile(model)  # PyTorch 2.0+
```

**To use:**
```python
# In any config file, set:
compile = True
```

### Advanced Compilation Options

For more control, update the compilation section in `train.py`:

```python
# Advanced torch.compile configuration
if compile:
    print("Compiling the model with optimizations...")

    # Compilation modes:
    # - 'default': Balanced (recommended)
    # - 'reduce-overhead': Minimize Python overhead
    # - 'max-autotune': Maximum optimization (slower compile, faster runtime)
    compile_mode = 'default'

    # Backend options:
    # - 'inductor': Default, best performance (PyTorch 2.0+)
    # - 'aot_eager': Ahead-of-time compilation
    # - 'cudagraphs': CUDA graphs (very fast, less flexible)
    compile_backend = 'inductor'

    unoptimized_model = model
    model = torch.compile(
        model,
        mode=compile_mode,
        backend=compile_backend,
        fullgraph=False,  # Allow graph breaks for flexibility
        dynamic=False     # Assume static shapes for better optimization
    )

    print(f"  Mode: {compile_mode}")
    print(f"  Backend: {compile_backend}")
```

---

## Step 2: Benchmark Compilation

**Purpose:** Measure compilation speedup  
**Duration:** 20 minutes

### Create benchmark_compile.py

```python
# benchmark_compile.py
"""
Benchmark torch.compile speedup.
"""

import time
import torch
from model import GPT, GPTConfig

def benchmark_model(model, batch_size=4, seq_len=256, num_iters=50):
    """
    Benchmark model training speed.

    Args:
        model: GPT model
        batch_size: Batch size
        seq_len: Sequence length
        num_iters: Number of iterations

    Returns:
        float: Average time per iteration (ms)
    """
    device = next(model.parameters()).device

    # Create dummy data
    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(10):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Benchmark
    torch.cuda.synchronize()
    times = []

    for _ in range(num_iters):
        t0 = time.time()

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        times.append(time.time() - t0)

    return sum(times) / len(times) * 1000  # Convert to ms


if __name__ == '__main__':
    print("=" * 80)
    print("torch.compile Benchmark")
    print("=" * 80 + "\n")

    # Create model
    config = GPTConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=1024,
        vocab_size=50304,
        dropout=0.0,
        bias=False
    )

    print(f"Model: {config.n_layer} layers, {config.n_embd} dim")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        exit(0)

    device = 'cuda'

    # Baseline (no compilation)
    print("\n1. Baseline (no compilation):")
    model_baseline = GPT(config).to(device)
    time_baseline = benchmark_model(model_baseline, batch_size=8, seq_len=512, num_iters=50)
    print(f"   Average time: {time_baseline:.2f}ms per iteration")

    # Compiled
    print("\n2. With torch.compile:")
    model_compiled = GPT(config).to(device)
    print("   Compiling... (this takes ~1-2 minutes)")
    model_compiled = torch.compile(model_compiled)
    time_compiled = benchmark_model(model_compiled, batch_size=8, seq_len=512, num_iters=50)
    print(f"   Average time: {time_compiled:.2f}ms per iteration")

    # Results
    speedup = time_baseline / time_compiled
    print("\n" + "=" * 80)
    print(f"Speedup: {speedup:.2f}x")
    print("=" * 80)

    if speedup > 1.5:
        print("✓ Good speedup achieved!")
    else:
        print("⚠ Speedup lower than expected. Check PyTorch version (requires 2.0+)")
```

### Run Benchmark

```bash
python benchmark_compile.py
```

### Expected Output

```
================================================================================
torch.compile Benchmark
================================================================================

Model: 12 layers, 768 dim

1. Baseline (no compilation):
   Average time: 145.32ms per iteration

2. With torch.compile:
   Compiling... (this takes ~1-2 minutes)
   Average time: 68.45ms per iteration

================================================================================
Speedup: 2.12x
================================================================================
✓ Good speedup achieved!
```

---

## Step 3: Flash Attention

**Purpose:** Implement memory-efficient attention  
**Duration:** 45 minutes

### Understanding Flash Attention

Flash Attention (Dao et al., 2022):
- Memory-efficient: O(N) instead of O(N²)
- Fast: Fused kernel, fewer memory accesses
- Exact: Same output as standard attention

**Benefits:**
- 2-4x memory reduction
- 2-3x speed improvement
- Enables longer context lengths

**Two versions available:**

1. **PyTorch Native (PyTorch 2.0+):**
   - Built-in to PyTorch
   - Automatically used if available
   - Good performance

2. **Flash Attention 2.x (separate package):**
   - Even faster
   - Requires CUDA compilation
   - Optional but recommended

### Installation (Optional)

For Flash Attention 2.x package:

```bash
# Install flash-attn (requires CUDA)
pip install flash-attn --no-build-isolation

# Or for specific CUDA version:
# CUDA 11.8
pip install flash-attn==2.3.6 --no-build-isolation

# CUDA 12.1
pip install flash-attn==2.4.2 --no-build-isolation
```

**Note:** Installation can take 5-10 minutes (compiles CUDA code)

### Flash Attention Already Integrated!

Good news: Flash Attention support is already in `model.py` from Phase 2B!

The `CausalSelfAttention` class automatically:
1. Checks for PyTorch native Flash Attention
2. Tries to import flash_attn package
3. Falls back to manual attention if neither available

**To verify which you're using:**
```python
python -c "from model import GPT, GPTConfig; m = GPT(GPTConfig()); print(m.transformer.h[0].attn.flash, m.transformer.h[0].attn.use_flash_attn)"
```

---

## Step 4: Benchmark Flash Attention

**Purpose:** Measure Flash Attention impact  
**Duration:** 15 minutes

### Add to benchmark_compile.py

Add this function to `benchmark_compile.py`:

```python
def benchmark_attention(block_size=1024):
    """Benchmark different attention implementations."""
    print("\n" + "=" * 80)
    print("Attention Implementation Benchmark")
    print("=" * 80 + "\n")

    config = GPTConfig(
        n_layer=1,  # Single layer for isolated testing
        n_head=12,
        n_embd=768,
        block_size=block_size,
        vocab_size=50304,
        dropout=0.0
    )

    batch_size = 4
    device = 'cuda'

    # Test different implementations
    implementations = []

    # Standard attention (disable flash)
    model_std = GPT(config).to(device)
    model_std.transformer.h[0].attn.flash = False
    model_std.transformer.h[0].attn.use_flash_attn = False
    implementations.append(("Standard Attention", model_std))

    # PyTorch Flash Attention
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        model_pytorch_flash = GPT(config).to(device)
        model_pytorch_flash.transformer.h[0].attn.use_flash_attn = False
        model_pytorch_flash.transformer.h[0].attn.flash = True
        implementations.append(("PyTorch Flash Attention", model_pytorch_flash))

    # Flash Attention 2.x
    try:
        from flash_attn import flash_attn_func
        model_flash2 = GPT(config).to(device)
        model_flash2.transformer.h[0].attn.use_flash_attn = True
        implementations.append(("Flash Attention 2.x", model_flash2))
    except ImportError:
        print("Flash Attention 2.x not installed (optional)")

    # Benchmark each
    results = {}
    for name, model in implementations:
        print(f"Testing {name}...")
        time_ms = benchmark_model(model, batch_size=batch_size, seq_len=block_size, num_iters=50)
        results[name] = time_ms
        print(f"  Time: {time_ms:.2f}ms\n")

    # Compare
    print("=" * 80)
    print("Results:")
    baseline = results.get("Standard Attention")
    for name, time_ms in results.items():
        speedup = baseline / time_ms if baseline else 1.0
        print(f"  {name}: {time_ms:.2f}ms ({speedup:.2f}x)")
    print("=" * 80)


# Add to main:
if __name__ == '__main__':
    # ... existing code ...
    
    # Add attention benchmark
    benchmark_attention(block_size=1024)
```

### Run Extended Benchmark

```bash
python benchmark_compile.py
```

### Expected Output

```
================================================================================
torch.compile Benchmark
================================================================================
...
Speedup: 2.12x
================================================================================

================================================================================
Attention Implementation Benchmark
================================================================================

Testing Standard Attention...
  Time: 187.45ms

Testing PyTorch Flash Attention...
  Time: 95.23ms

Flash Attention 2.x not installed (optional)

================================================================================
Results:
  Standard Attention: 187.45ms (1.00x)
  PyTorch Flash Attention: 95.23ms (1.97x)
================================================================================
```

**If you installed Flash Attention 2.x:**
```
  Flash Attention 2.x: 68.34ms (2.74x)
```

---

## Step 5: Combined Optimization Test

**Purpose:** Test compile + Flash Attention together  
**Duration:** 10 minutes

### Create test_optimizations.py

```python
# test_optimizations.py
"""
Test combined optimizations.
"""

import time
import torch
from model import GPT, GPTConfig


def quick_benchmark(model, name, iterations=20):
    """Quick benchmark of model."""
    device = next(model.parameters()).device
    
    x = torch.randint(0, model.config.vocab_size, (4, 512), device=device)
    y = torch.randint(0, model.config.vocab_size, (4, 512), device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Warmup
    for _ in range(5):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Time
    torch.cuda.synchronize()
    t0 = time.time()
    
    for _ in range(iterations):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    avg_time = (time.time() - t0) / iterations * 1000
    
    print(f"{name}: {avg_time:.2f}ms per iteration")
    return avg_time


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA required for this test")
        exit(0)
    
    print("=" * 60)
    print("Combined Optimization Test")
    print("=" * 60 + "\n")
    
    config = GPTConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=1024,
        vocab_size=50304,
        dropout=0.0
    )
    
    results = {}
    
    # Baseline
    print("1. Baseline (no optimizations)")
    model = GPT(config).cuda()
    results['baseline'] = quick_benchmark(model, "  Baseline")
    print()
    
    # Compiled only
    print("2. With torch.compile")
    model = GPT(config).cuda()
    model = torch.compile(model)
    results['compile'] = quick_benchmark(model, "  Compiled")
    print()
    
    # Flash Attention (already in model by default)
    print("3. With Flash Attention")
    print("  (PyTorch native Flash Attention)")
    model = GPT(config).cuda()
    results['flash'] = quick_benchmark(model, "  Flash")
    print()
    
    # Both!
    print("4. With both compile + Flash Attention")
    model = GPT(config).cuda()
    model = torch.compile(model)
    results['both'] = quick_benchmark(model, "  Both")
    print()
    
    # Summary
    print("=" * 60)
    print("Summary:")
    baseline = results['baseline']
    for name, time_ms in results.items():
        speedup = baseline / time_ms
        print(f"  {name}: {speedup:.2f}x speedup")
    print("=" * 60)
```

### Run Test

```bash
python test_optimizations.py
```

### Expected Output

```
============================================================
Combined Optimization Test
============================================================

1. Baseline (no optimizations)
  Baseline: 145.67ms per iteration

2. With torch.compile
  Compiled: 72.34ms per iteration

3. With Flash Attention
  (PyTorch native Flash Attention)
  Flash: 93.45ms per iteration

4. With both compile + Flash Attention
  Both: 51.23ms per iteration

============================================================
Summary:
  baseline: 1.00x speedup
  compile: 2.01x speedup
  flash: 1.56x speedup
  both: 2.84x speedup
============================================================
```

**Key takeaway:** Optimizations stack! Combined speedup is multiplicative.

---

## Step 6: Enable in Your Training

**Purpose:** Use optimizations in real training  
**Duration:** 5 minutes

### Update Your Config

In any config file (e.g., `config/train_shakespeare_char.py`):

```python
# Enable optimizations
compile = True  # PyTorch 2.0+ required

# Use BF16 if available (works better with compile)
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
```

### Train with Optimizations

```bash
python train.py config/train_shakespeare_char.py
```

**You should see:**
```
Compiling the model... (takes ~1 minute)
Using PyTorch native Flash Attention
...
iter 10: loss 3.8234, time 65.32ms, mfu 28.23%
```

**Compare to before:**
- Before: ~125ms per iteration, MFU ~18%
- After: ~65ms per iteration, MFU ~28%
- **~2x faster!**

---

## Troubleshooting

### Issue: "torch.compile not available"

**Problem:** PyTorch version < 2.0

**Solutions:**
1. Upgrade PyTorch: `pip install --upgrade torch`
2. Or disable: `compile = False` in config

### Issue: Compilation fails with error

**Problem:** Some operations not supported

**Solutions:**
1. Try different mode: `compile_mode = 'reduce-overhead'`
2. Enable graph breaks: `fullgraph = False`
3. Disable if persistent: `compile = False`

### Issue: Flash Attention not working

**Check which version you have:**
```python
python -c "import torch; print(hasattr(torch.nn.functional, 'scaled_dot_product_attention'))"
```

**Solutions:**
- `True`: PyTorch Flash Attention available (good!)
- `False`: Upgrade to PyTorch 2.0+
- For Flash Attention 2.x: Install flash-attn package

### Issue: First iteration very slow

**Expected!** 
- Compilation happens on first iteration
- Takes ~1-2 minutes
- Subsequent iterations are fast
- This is normal behavior

---

## Phase 6A Complete!

### What You've Built

✅ **Speed Optimizations**:
- PyTorch 2.0 compilation integration (2-4x speedup)
- Flash Attention support (2-3x speed + memory)
- Advanced compilation options
- Benchmarking scripts
- Combined optimization testing
- Training integration

### Key Files Created/Modified

- `train.py`: Enhanced with compilation options
- `benchmark_compile.py`: Compilation benchmarks
- `test_optimizations.py`: Combined optimization tests

### Performance Improvements

**Typical speedups:**
- torch.compile alone: 2.0-2.5x
- Flash Attention alone: 1.5-2.0x
- Combined: 2.5-4.0x
- MFU improvement: 18% → 25-30%

**Memory savings (Flash Attention):**
- Standard attention: O(seq_len²) memory
- Flash Attention: O(seq_len) memory
- Enables 2-4x longer contexts

---

## Next Substage

**Proceed to Phase 6B: Memory Optimization**

Now that we have speed optimizations, we'll add memory optimizations:
- Gradient checkpointing (trade compute for memory)
- Memory profiling tools
- Memory benchmarks
- 30-50% memory savings

**Estimated time:** 1.5 hours

This lets you train larger models on the same hardware!

---

**Phase 6A Character Count:** ~16,200 characters
