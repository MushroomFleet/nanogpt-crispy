# Phase 2A: Foundation Components

## Substage Overview

- **Parent Phase:** Phase 2: Core Model Architecture
- **Substage:** A of 3 (Foundation)
- **Goal:** Build the basic building blocks of the GPT model
- **Prerequisites:** Phase 1 completed (project setup)
- **Estimated Duration:** 1.5 hours
- **Key Deliverables:**
  - GPTConfig dataclass for model configuration
  - LayerNorm implementation with optional bias
  - MLP (feed-forward network) implementation
  - Basic component tests

---

## What We're Building

In this substage, we'll create three fundamental components that every transformer needs:

1. **Configuration System** - Clean way to define model hyperparameters
2. **Layer Normalization** - Stabilizes training in deep networks
3. **Feed-Forward Network** - Adds non-linearity and capacity

These are the simpler components. In the next substage (2B), we'll tackle the complex attention mechanism.

---

## Architecture Overview

Before we start, here's where these components fit in the GPT architecture:

```
GPT Architecture (Decoder-Only Transformer)

Input: Token IDs [batch_size, seq_len]
  ↓
Token Embedding + Position Embedding
  ↓
Dropout
  ↓
┌──────────────────────────────────────┐
│ Transformer Block (repeated n_layer)│
│                                      │
│  Input                               │
│   ↓                                  │
│  LayerNorm  ← We build this today   │
│   ↓                                  │
│  CausalSelfAttention (Phase 2B)     │
│   ↓                                  │
│  + (residual connection)             │
│   ↓                                  │
│  LayerNorm  ← We build this today   │
│   ↓                                  │
│  MLP        ← We build this today   │
│   ↓                                  │
│  + (residual connection)             │
└──────────────────────────────────────┘
  ↓
LayerNorm
  ↓
Linear Head
  ↓
Output Logits
```

---

## Step 1: GPTConfig Dataclass

**Purpose:** Define all model hyperparameters in a clean structure  
**Duration:** 10 minutes

### Create model.py

Create or edit `model.py`:

```python
# model.py
"""
Full definition of a GPT Language Model.

References:
1) GPT-2 paper: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
2) OpenAI GPT-2 code: https://github.com/openai/gpt-2/blob/master/src/model.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """
    Configuration class for GPT model.

    Default values are for GPT-2 small (124M parameters).
    """
    # Model architecture
    block_size: int = 1024      # Maximum sequence length
    vocab_size: int = 50304     # Vocabulary size (50257 rounded up for efficiency)
    n_layer: int = 12           # Number of transformer blocks
    n_head: int = 12            # Number of attention heads
    n_embd: int = 768           # Embedding dimension
    dropout: float = 0.0        # Dropout probability
    bias: bool = True           # Use bias in Linear and LayerNorm layers

    # Common configurations
    @classmethod
    def gpt2_small(cls):
        """GPT-2 Small: 124M parameters"""
        return cls(n_layer=12, n_head=12, n_embd=768)

    @classmethod
    def gpt2_medium(cls):
        """GPT-2 Medium: 350M parameters"""
        return cls(n_layer=24, n_head=16, n_embd=1024)

    @classmethod
    def gpt2_large(cls):
        """GPT-2 Large: 774M parameters"""
        return cls(n_layer=36, n_head=20, n_embd=1280)

    @classmethod
    def gpt2_xl(cls):
        """GPT-2 XL: 1558M parameters"""
        return cls(n_layer=48, n_head=25, n_embd=1600)
```

### Understanding the Config

**Key Fields Explained:**

- **block_size**: Maximum context length - how many tokens the model can "see" at once (GPT-2 uses 1024)
- **vocab_size**: Number of unique tokens in our vocabulary (50257 for GPT-2 BPE, rounded to 50304 for efficiency)
- **n_layer**: How many transformer blocks to stack (more = more powerful but slower)
- **n_head**: Number of parallel attention heads in each block
- **n_embd**: Size of the embedding vectors (higher = more capacity)
- **dropout**: Dropout rate for regularization (0.0 = no dropout)
- **bias**: Whether to use bias terms in layers

**Why round vocab_size to 50304?**
- 50304 = 64 × 786 (divisible by powers of 2)
- Better GPU utilization (GPUs like powers of 2)
- Minimal overhead (only 47 unused tokens)

### Test It

Add a quick test at the bottom of `model.py`:

```python
if __name__ == '__main__':
    # Quick test
    config = GPTConfig.gpt2_small()
    print(f"Config created: {config.n_layer} layers, {config.n_embd} dim")
```

Run:
```bash
python model.py
```

Expected output:
```
Config created: 12 layers, 768 dim
```

### Verification Checklist
- [ ] GPTConfig dataclass created in model.py
- [ ] All fields have correct types and defaults
- [ ] Factory methods for GPT-2 variants included
- [ ] Quick test runs successfully

---

## Step 2: LayerNorm Implementation

**Purpose:** Implement Layer Normalization for stable training  
**Duration:** 10 minutes

### Add LayerNorm Class

Add this class to `model.py` (after GPTConfig):

```python
class LayerNorm(nn.Module):
    """
    LayerNorm with optional bias.

    PyTorch's nn.LayerNorm doesn't support removing bias without subclassing.
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```

### Understanding LayerNorm

**What it does:**
- Normalizes across the feature dimension (not batch like BatchNorm)
- Formula: `y = (x - mean) / sqrt(variance + eps) × weight + bias`
- Stabilizes training by keeping activations in a reasonable range

**Why custom implementation?**
- GPT-2 doesn't use bias in LayerNorm
- PyTorch's built-in `nn.LayerNorm` always includes bias
- Our implementation respects the `bias` config parameter

**When is it used?**
- Before attention in each transformer block
- Before the feed-forward network (MLP) in each block
- After all transformer blocks (final layer norm)

### Test LayerNorm

Add test to the bottom of `model.py`:

```python
if __name__ == '__main__':
    # Test LayerNorm
    ln = LayerNorm(768, bias=True)
    x = torch.randn(2, 10, 768)  # (batch, seq_len, embedding_dim)
    y = ln(x)
    print(f"LayerNorm test: input {x.shape} -> output {y.shape}")
    print(f"  Mean: {y.mean():.6f}, Std: {y.std():.6f}")
```

Run and you should see output like:
```
LayerNorm test: input torch.Size([2, 10, 768]) -> output torch.Size([2, 10, 768])
  Mean: 0.000123, Std: 1.000456
```

### Verification Checklist
- [ ] LayerNorm class added to model.py
- [ ] Optional bias parameter works
- [ ] Test shows normalized output (mean ≈ 0, std ≈ 1)

---

## Step 3: MLP (Feed-Forward Network)

**Purpose:** Implement the position-wise feed-forward network  
**Duration:** 15 minutes

### Add MLP Class

Add this class to `model.py` (after LayerNorm):

```python
class MLP(nn.Module):
    """
    Multi-Layer Perceptron (feed-forward network) used in transformer blocks.

    Architecture: Linear -> GELU -> Linear -> Dropout
    Hidden dimension is 4x the embedding dimension (GPT-2 convention).
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)      # Project up: (B, T, n_embd) -> (B, T, 4*n_embd)
        x = self.gelu(x)      # Non-linearity
        x = self.c_proj(x)    # Project down: (B, T, 4*n_embd) -> (B, T, n_embd)
        x = self.dropout(x)
        return x
```

### Understanding the MLP

**Architecture:**
```
Input: [batch, seq_len, 768]
  ↓
Linear (expand 4x): [batch, seq_len, 3072]
  ↓
GELU activation
  ↓
Linear (project back): [batch, seq_len, 768]
  ↓
Dropout
  ↓
Output: [batch, seq_len, 768]
```

**Key Points:**

1. **4x Expansion**: We expand to `4 × n_embd` then project back
   - Empirical choice from GPT-2 paper
   - Adds capacity without being too expensive
   - Larger models sometimes use different ratios

2. **GELU Activation**: Gaussian Error Linear Unit
   - Smoother than ReLU
   - GPT-2's choice for better performance
   - Formula: `GELU(x) = x × Φ(x)` where Φ is standard Gaussian CDF

3. **Applied Per-Position**: Same weights for all positions in sequence
   - Each position processed independently
   - No interaction between positions (that's attention's job)

### Test MLP

Add test to `model.py`:

```python
if __name__ == '__main__':
    # Test MLP
    config = GPTConfig(n_embd=768, dropout=0.1, bias=True)
    mlp = MLP(config)
    x = torch.randn(2, 10, 768)  # (batch, seq_len, embedding_dim)
    y = mlp(x)
    print(f"MLP test: input {x.shape} -> output {y.shape}")
    
    # Count parameters
    n_params = sum(p.numel() for p in mlp.parameters())
    print(f"  MLP parameters: {n_params:,}")
```

Expected output:
```
MLP test: input torch.Size([2, 10, 768]) -> output torch.Size([2, 10, 768])
  MLP parameters: 4,722,432
```

### Verification Checklist
- [ ] MLP class added to model.py
- [ ] 4x expansion implemented correctly
- [ ] GELU activation used
- [ ] Dropout applied
- [ ] Test shows correct input/output shapes

---

## Complete Test Suite

Let's create a comprehensive test for all our components so far.

### Create test_components.py

```python
# test_components.py
"""
Test the foundation components we built in Phase 2A.
"""

import torch
from model import GPTConfig, LayerNorm, MLP


def test_config():
    """Test GPTConfig creation."""
    print("Testing GPTConfig...")
    
    # Default config
    config = GPTConfig()
    assert config.n_layer == 12
    assert config.n_embd == 768
    print("  ✓ Default config works")
    
    # Custom config
    config = GPTConfig(n_layer=6, n_embd=384, dropout=0.2)
    assert config.n_layer == 6
    assert config.n_embd == 384
    assert config.dropout == 0.2
    print("  ✓ Custom config works")
    
    # Factory methods
    config = GPTConfig.gpt2_large()
    assert config.n_layer == 36
    assert config.n_embd == 1280
    print("  ✓ Factory methods work")
    print()


def test_layernorm():
    """Test LayerNorm."""
    print("Testing LayerNorm...")
    
    # With bias
    ln = LayerNorm(768, bias=True)
    assert ln.bias is not None
    print("  ✓ Creates with bias")
    
    # Without bias
    ln = LayerNorm(768, bias=False)
    assert ln.bias is None
    print("  ✓ Creates without bias")
    
    # Forward pass
    x = torch.randn(4, 16, 768)
    y = ln(x)
    assert y.shape == x.shape
    assert abs(y.mean().item()) < 0.01  # Mean should be close to 0
    assert abs(y.std().item() - 1.0) < 0.1  # Std should be close to 1
    print(f"  ✓ Forward pass: mean={y.mean():.6f}, std={y.std():.6f}")
    print()


def test_mlp():
    """Test MLP."""
    print("Testing MLP...")
    
    config = GPTConfig(n_embd=768, dropout=0.0, bias=True)
    mlp = MLP(config)
    
    # Check layers exist
    assert hasattr(mlp, 'c_fc')
    assert hasattr(mlp, 'gelu')
    assert hasattr(mlp, 'c_proj')
    print("  ✓ All layers present")
    
    # Forward pass
    x = torch.randn(4, 16, 768)
    y = mlp(x)
    assert y.shape == x.shape
    print(f"  ✓ Forward pass: {x.shape} -> {y.shape}")
    
    # Check 4x expansion
    assert mlp.c_fc.out_features == 4 * config.n_embd
    assert mlp.c_proj.in_features == 4 * config.n_embd
    print("  ✓ 4x expansion confirmed")
    
    # Count parameters
    n_params = sum(p.numel() for p in mlp.parameters())
    expected = (768 * 3072 + 3072) + (3072 * 768 + 768)  # Two linear layers
    assert n_params == expected
    print(f"  ✓ Parameter count correct: {n_params:,}")
    print()


def test_gradient_flow():
    """Test that gradients flow through components."""
    print("Testing gradient flow...")
    
    config = GPTConfig(n_embd=768, dropout=0.0)
    
    # LayerNorm
    ln = LayerNorm(768, bias=True)
    x = torch.randn(2, 10, 768, requires_grad=True)
    y = ln(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    print("  ✓ Gradients flow through LayerNorm")
    
    # MLP
    mlp = MLP(config)
    x = torch.randn(2, 10, 768, requires_grad=True)
    y = mlp(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    print("  ✓ Gradients flow through MLP")
    print()


def test_cuda_if_available():
    """Test on CUDA if available."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return
    
    print("Testing on CUDA...")
    
    config = GPTConfig(n_embd=768, dropout=0.0)
    
    # LayerNorm on GPU
    ln = LayerNorm(768, bias=True).cuda()
    x = torch.randn(2, 10, 768).cuda()
    y = ln(x)
    assert y.is_cuda
    print("  ✓ LayerNorm works on GPU")
    
    # MLP on GPU
    mlp = MLP(config).cuda()
    x = torch.randn(2, 10, 768).cuda()
    y = mlp(x)
    assert y.is_cuda
    print("  ✓ MLP works on GPU")
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("Phase 2A: Foundation Components Tests")
    print("=" * 60 + "\n")
    
    test_config()
    test_layernorm()
    test_mlp()
    test_gradient_flow()
    test_cuda_if_available()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
```

### Run Tests

```bash
python test_components.py
```

### Expected Output

```
============================================================
Phase 2A: Foundation Components Tests
============================================================

Testing GPTConfig...
  ✓ Default config works
  ✓ Custom config works
  ✓ Factory methods work

Testing LayerNorm...
  ✓ Creates with bias
  ✓ Creates without bias
  ✓ Forward pass: mean=0.000123, std=1.000456

Testing MLP...
  ✓ All layers present
  ✓ Forward pass: torch.Size([4, 16, 768]) -> torch.Size([4, 16, 768])
  ✓ 4x expansion confirmed
  ✓ Parameter count correct: 4,722,432

Testing gradient flow...
  ✓ Gradients flow through LayerNorm
  ✓ Gradients flow through MLP

Testing on CUDA...
  ✓ LayerNorm works on GPU
  ✓ MLP works on GPU

============================================================
All tests passed! ✓
============================================================
```

---

## Troubleshooting

### Issue: Import errors

**Problem:** `ModuleNotFoundError` when running tests

**Solution:**
1. Make sure you're in the project root directory
2. Check that `model.py` exists
3. Ensure virtual environment is activated

### Issue: CUDA out of memory (during GPU test)

**Problem:** GPU runs out of memory

**Solution:**
- Normal for small GPUs - tests use small tensors
- If persistent, skip GPU test (comment out `test_cuda_if_available()`)

### Issue: Gradient flow test fails

**Problem:** `x.grad is None`

**Solution:**
- Ensure `requires_grad=True` when creating tensors
- Check that loss.backward() is called
- Verify x is actually used in the computation

---

## Phase 2A Complete!

### What You've Built

✅ **Foundation Components**:
- GPTConfig dataclass with factory methods
- Custom LayerNorm with optional bias
- MLP (feed-forward network) with 4x expansion
- Complete test suite validating all components

### Key Files

- `model.py`: Contains GPTConfig, LayerNorm, MLP
- `test_components.py`: Comprehensive tests

### Component Statistics

**LayerNorm:**
- Parameters: 2 × n_embd (weight + bias)
- For n_embd=768: 1,536 parameters

**MLP:**
- Parameters: ~4.7M for n_embd=768
- Expansion: 768 → 3072 → 768
- Activation: GELU (smoother than ReLU)

---

## Next Substage

**Proceed to Phase 2B: Attention Mechanism**

In the next substage, we'll build the most complex and important component:
- Multi-head causal self-attention
- Query, Key, Value projections
- Attention masking for autoregressive generation
- Flash Attention support (optional)

**Estimated time:** 2 hours

This is the core innovation that makes transformers work!

---

**Phase 2A Character Count:** ~15,400 characters
