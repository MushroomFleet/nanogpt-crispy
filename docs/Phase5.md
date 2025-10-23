# Phase 5: Inference and Generation

## Phase Overview

**Goal:** Implement text generation with various sampling strategies

**Prerequisites:**
- Phase 1-4 completed (environment, model, data, training)
- At least one trained model checkpoint
- Understanding of sampling methods

**Estimated Duration:** 2-3 hours

**Key Deliverables:**
- Complete `sample.py` generation script
- Checkpoint loading for inference
- Autoregressive generation implementation
- Temperature-based sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Interactive generation interface
- Batch generation support

---

## Understanding Text Generation

```
Autoregressive Generation Process

Input: "Once upon a"
  ↓
Encode: [7454, 2402, 257]
  ↓
┌─────────────────────────────┐
│ Generation Loop (repeat N)  │
│                              │
│ 1. Forward pass through GPT │
│    → Get logits for next    │
│                              │
│ 2. Apply temperature        │
│    → Scale logits           │
│                              │
│ 3. Apply top-k filter       │
│    → Keep k most likely     │
│                              │
│ 4. Softmax → probabilities  │
│                              │
│ 5. Sample next token        │
│    → Multinomial sampling   │
│                              │
│ 6. Append to context        │
│    → Update input           │
└─────────────────────────────┘
  ↓
Decode: [7454, 2402, 257, 640, 11, 612, 373, 257, ...]
  ↓
Output: "Once upon a time, there was a princess..."
```

**Key Concepts:**

- **Autoregressive**: Generate one token at a time, conditioning on all previous tokens
- **Temperature**: Controls randomness (low = deterministic, high = creative)
- **Top-k**: Only sample from k most likely tokens
- **Top-p**: Sample from tokens whose cumulative probability is p

---

## Step 1: Basic Sample Script Structure

**Purpose:** Create the main generation script
**Duration:** 20 minutes

Create `sample.py`:

```python
# sample.py
"""
Sample from a trained GPT model.

Usage:
    # Generate from checkpoint
    python sample.py --out_dir=out-shakespeare-char

    # Custom generation
    python sample.py --out_dir=out-shakespeare-char --start="ROMEO:" --num_samples=5

    # Interactive mode
    python sample.py --out_dir=out-shakespeare-char --interactive
"""

import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration
out_dir = 'out'  # Directory containing checkpoint
start = "\n"  # Starting prompt (or "FILE:prompt.txt" to load from file)
num_samples = 1  # Number of samples to generate
max_new_tokens = 500  # Number of tokens to generate
temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random
top_k = 200  # Retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda'  # 'cpu', 'cuda', 'cuda:0', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # Use PyTorch 2.0 compile for faster generation
# -----------------------------------------------------------------------------

# Parse command line arguments
import sys
for arg in sys.argv[1:]:
    if '=' in arg:
        key, val = arg.split('=')
        key = key.replace('--', '')
        try:
            val = eval(val)
        except (SyntaxError, NameError):
            pass
        if key in globals():
            print(f"Setting {key} = {val}")
            globals()[key] = val

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load checkpoint
print(f"Loading checkpoint from {out_dir}")
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)

# Create model
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']

# Fix the keys if needed
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)

if compile:
    print("Compiling model...")
    model = torch.compile(model)

print(f"Model loaded: {gptconf.n_layer} layers, {gptconf.n_embd} dim")
```

### Explanation

**Command-line Argument Parsing:**
```python
# Override defaults from command line
python sample.py --temperature=0.9 --num_samples=3
```

**Checkpoint Loading:**
- Loads model architecture from `model_args`
- Loads trained weights from `model`
- Handles both compiled and uncompiled models

**Device and Precision:**
- Uses same dtype as training for consistency
- Mixed precision speeds up generation
- Falls back to float32 on CPU

### Verification
- [ ] sample.py created
- [ ] Checkpoint loading implemented
- [ ] Model initialization works

---

## Step 2: Tokenization Setup

**Purpose:** Set up encoding and decoding
**Duration:** 15 minutes

Add to `sample.py`:

```python
# Load tokenizer
load_meta = False
if 'config' in checkpoint and 'dataset' in checkpoint['config']:
    # Try to load meta from dataset
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    # Check if we need character-level encoding
    if 'stoi' in meta:
        # Character-level dataset
        print("Character-level tokenization")
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # GPT-2 BPE tokenization
        print("GPT-2 BPE tokenization")
        enc = tiktoken.get_encoding(meta.get('tokenizer', 'gpt2'))
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
else:
    # Default to GPT-2 tokenization
    print("No meta.pkl found, using GPT-2 BPE tokenization")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# Encode the starting prompt
if start.startswith('FILE:'):
    # Load prompt from file
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

print(f"Starting prompt: '{start}'")
print(f"Encoded to {len(start_ids)} tokens")
```

### Explanation

**Tokenization Support:**
1. **Character-level**: For datasets prepared with character-level encoding
2. **GPT-2 BPE**: Standard for most use cases
3. **Auto-detection**: Reads from dataset's `meta.pkl`

**Prompt Loading:**
- Direct string: `--start="Once upon a time"`
- From file: `--start=FILE:prompt.txt`
- Default: `"\n"` (newline, lets model start freely)

**Encoding:**
```python
start_ids = encode("Hello")  # [15496]
x = torch.tensor(start_ids, device=device)[None, ...]  # [1, 1] (batch=1, seq=1)
```

### Verification
- [ ] Tokenization setup added
- [ ] Both character and BPE supported
- [ ] Prompt encoding works

---

## Step 3: Generation Function

**Purpose:** Implement the core generation loop
**Duration:** 45 minutes

Add to `sample.py`:

```python
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    """
    Generate text autoregressively.

    Args:
        model: GPT model
        idx: Starting context [batch_size, seq_len]
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top k tokens
        top_p: If set, use nucleus sampling

    Returns:
        Generated token indices [batch_size, seq_len + max_new_tokens]
    """
    for _ in range(max_new_tokens):
        # Crop context to block_size if needed
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]

        # Forward pass
        with ctx:
            logits, _ = model(idx_cond)

        # Get logits for last position
        logits = logits[:, -1, :]  # [batch_size, vocab_size]

        # Apply temperature
        logits = logits / temperature

        # Apply top-k sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Apply top-p (nucleus) sampling
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Set logits to -inf for removed tokens
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Sample from distribution
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

### Explanation

**Temperature Sampling:**
```python
logits = logits / temperature

# temperature = 0.1: Very deterministic, always picks most likely
# temperature = 1.0: Standard sampling
# temperature = 2.0: Very creative, samples from broader distribution
```

**Top-k Sampling:**
```python
# Only keep top k most likely tokens
v, _ = torch.topk(logits, k)
logits[logits < v[:, [-1]]] = -float('Inf')

# Example with k=3:
# Original: [0.5, 0.3, 0.15, 0.04, 0.01]
# After:    [0.5, 0.3, 0.15, -inf, -inf]
# Renormalized: [0.526, 0.316, 0.158, 0.0, 0.0]
```

**Top-p (Nucleus) Sampling:**
```python
# Sample from smallest set of tokens whose cumulative prob >= p
# Example with p=0.9:
# Probs: [0.5, 0.3, 0.15, 0.04, 0.01]
# Cumulative: [0.5, 0.8, 0.95, ...]
# Keep first 3 tokens (0.5 + 0.3 + 0.15 = 0.95 > 0.9)
```

**Which to use?**
- **Temperature alone**: Good for most cases
- **Top-k (50-200)**: Prevents rare/nonsensical tokens
- **Top-p (0.9-0.95)**: Adaptive cutoff (better than top-k)
- **Combination**: Often best (e.g., temp=0.8, top_k=200)

### Verification
- [ ] Generate function implemented
- [ ] Temperature scaling works
- [ ] Top-k filtering works
- [ ] Top-p (nucleus) sampling works

---

## Step 4: Main Generation Loop

**Purpose:** Generate multiple samples
**Duration:** 15 minutes

Add to `sample.py`:

```python
# Generate samples
print(f"\nGenerating {num_samples} samples...\n")
print("=" * 80)

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            # Generate
            y = generate(model, x, max_new_tokens, temperature=temperature, top_k=top_k)

            # Decode and print
            output = decode(y[0].tolist())
            print(output)
            print("=" * 80)
```

### Example Usage

```bash
# Generate from Shakespeare model
python sample.py --out_dir=out-shakespeare-char --num_samples=3 --temperature=0.8

# Start with custom prompt
python sample.py --out_dir=out-shakespeare-char --start="ROMEO:" --max_new_tokens=200

# Very creative generation
python sample.py --out_dir=out-shakespeare-char --temperature=1.5 --top_k=100

# Deterministic generation (greedy)
python sample.py --out_dir=out-shakespeare-char --temperature=0.1 --top_k=1
```

### Expected Output

```
Loading checkpoint from out-shakespeare-char
Model loaded: 6 layers, 384 dim
GPT-2 BPE tokenization
Starting prompt: '
'
Encoded to 1 tokens

Generating 3 samples...

================================================================================

ROMEO:
Come, what is your counsel?

JULIET:
I cannot speak of such a thing.

ROMEO:
Yet I will tell thee.

JULIET:
What sayst thou?

ROMEO:
If you be not too rashly made,
You must not speak of this.

================================================================================

DUKE OF EXETER:
What says the king?

KING HENRY VI:
Let him be brought forth, and let him speak.

DUKE OF GLOUCESTER:
My lord, I know not what you mean by this.

================================================================================

First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

================================================================================
```

### Verification
- [ ] Multiple samples generated
- [ ] Output is coherent
- [ ] Different samples show variety
- [ ] Text matches training data style

---

## Step 5: Interactive Mode

**Purpose:** Create an interactive generation interface
**Duration:** 30 minutes

Add to `sample.py`:

```python
def interactive_mode():
    """
    Interactive generation mode.

    User enters prompts, model generates continuations.
    """
    print("\n" + "=" * 80)
    print("Interactive Generation Mode")
    print("=" * 80)
    print("Enter a prompt and press Enter to generate.")
    print("Type 'quit' or 'exit' to end.")
    print("=" * 80 + "\n")

    while True:
        try:
            # Get prompt from user
            prompt = input("Prompt: ")

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not prompt.strip():
                prompt = "\n"  # Default to newline

            # Encode prompt
            start_ids = encode(prompt)
            x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

            # Generate
            print(f"\nGenerating ({max_new_tokens} tokens, temp={temperature}, top_k={top_k})...\n")

            with torch.no_grad():
                with ctx:
                    y = generate(model, x, max_new_tokens, temperature=temperature, top_k=top_k)

            # Decode and print
            output = decode(y[0].tolist())
            print(output)
            print("\n" + "-" * 80 + "\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


# Check if interactive mode requested
interactive = '--interactive' in sys.argv or '-i' in sys.argv

if interactive:
    interactive_mode()
else:
    # Standard batch generation (code from Step 4)
    print(f"\nGenerating {num_samples} samples...\n")
    print("=" * 80)

    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = generate(model, x, max_new_tokens, temperature=temperature, top_k=top_k)
                output = decode(y[0].tolist())
                print(output)
                print("=" * 80)
```

### Interactive Mode Usage

```bash
# Start interactive mode
python sample.py --out_dir=out-shakespeare-char --interactive

# Or with custom parameters
python sample.py --out_dir=out-shakespeare-char -i --temperature=0.9 --max_new_tokens=300
```

### Example Session

```
================================================================================
Interactive Generation Mode
================================================================================
Enter a prompt and press Enter to generate.
Type 'quit' or 'exit' to end.
================================================================================

Prompt: HAMLET:
Generating (500 tokens, temp=0.8, top_k=200)...

HAMLET:
To be, or not to be: that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them?

--------------------------------------------------------------------------------

Prompt: Once upon a time
Generating (500 tokens, temp=0.8, top_k=200)...

Once upon a time there lived a king who had three daughters. The eldest was very beautiful, but the youngest was the most beautiful of all. One day the king said to his daughters: "I am going to give you each a task to perform."

--------------------------------------------------------------------------------

Prompt: quit
Goodbye!
```

### Verification
- [ ] Interactive mode works
- [ ] Can enter multiple prompts
- [ ] Generation is fast enough
- [ ] Can exit cleanly

---

## Step 6: Batch Generation

**Purpose:** Generate multiple samples in parallel
**Duration:** 20 minutes

Add to `sample.py`:

```python
def batch_generate(prompts, max_new_tokens, temperature=1.0, top_k=None):
    """
    Generate from multiple prompts in parallel.

    Args:
        prompts: List of starting prompts (strings)
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering

    Returns:
        List of generated texts
    """
    # Encode all prompts
    encoded = [encode(prompt) for prompt in prompts]

    # Pad to same length
    max_len = max(len(ids) for ids in encoded)
    padded = []
    for ids in encoded:
        if len(ids) < max_len:
            ids = [0] * (max_len - len(ids)) + ids  # Pad left
        padded.append(ids)

    # Create batch tensor
    x = torch.tensor(padded, dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        with ctx:
            y = generate(model, x, max_new_tokens, temperature=temperature, top_k=top_k)

    # Decode all outputs
    outputs = []
    for i in range(y.size(0)):
        output = decode(y[i].tolist())
        outputs.append(output)

    return outputs


# Example usage for batch generation
def batch_generation_demo():
    """Demonstrate batch generation."""
    prompts = [
        "ROMEO:",
        "JULIET:",
        "HAMLET:",
    ]

    print(f"\nBatch generating {len(prompts)} samples...\n")

    outputs = batch_generate(prompts, max_new_tokens=200, temperature=0.8, top_k=200)

    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"Sample {i+1}:")
        print(output)
        print("=" * 80)
```

### Batch Generation Usage

```python
# In sample.py, add option for batch mode
batch_mode = '--batch' in sys.argv

if batch_mode:
    batch_generation_demo()
```

Run:
```bash
python sample.py --out_dir=out-shakespeare-char --batch
```

### Benefits of Batch Generation

- **Faster**: Process multiple prompts in parallel
- **Efficient**: Better GPU utilization
- **Useful for**: Generating multiple variations, A/B testing prompts

### Verification
- [ ] Batch generation implemented
- [ ] Multiple prompts processed together
- [ ] Outputs are correct

---

## Step 7: Advanced Sampling Options

**Purpose:** Add more sophisticated sampling strategies
**Duration:** 20 minutes

Add to `sample.py`:

```python
def sample_with_penalties(model, idx, max_new_tokens, temperature=1.0, top_k=None,
                          repetition_penalty=1.0, frequency_penalty=0.0):
    """
    Generate with repetition and frequency penalties.

    Args:
        model: GPT model
        idx: Starting context
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        repetition_penalty: Penalty for repeating tokens (> 1.0 = discourage)
        frequency_penalty: Penalty based on token frequency in context

    Returns:
        Generated token indices
    """
    token_counts = {}  # Track token frequencies

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]

        with ctx:
            logits, _ = model(idx_cond)

        logits = logits[:, -1, :]

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id, count in token_counts.items():
                # Penalize tokens that already appeared
                logits[0, token_id] /= repetition_penalty

        # Apply frequency penalty
        if frequency_penalty != 0.0:
            for token_id, count in token_counts.items():
                # Penalize based on frequency
                logits[0, token_id] -= frequency_penalty * count

        # Temperature
        logits = logits / temperature

        # Top-k
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Sample
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # Update token counts
        token_id = idx_next.item()
        token_counts[token_id] = token_counts.get(token_id, 0) + 1

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

### Explanation

**Repetition Penalty:**
- Discourages repeating the same tokens
- `> 1.0`: Penalize repetitions (e.g., 1.2)
- `= 1.0`: No penalty (default)
- Useful for preventing loops like "and and and and..."

**Frequency Penalty:**
- Penalizes tokens based on how often they appeared
- Encourages diversity in generated text
- OpenAI GPT-3 API uses this

### Usage Example

```python
# Generate with repetition penalty
y = sample_with_penalties(
    model, x, max_new_tokens=500,
    temperature=0.8,
    top_k=200,
    repetition_penalty=1.2,
    frequency_penalty=0.1
)
```

### Verification
- [ ] Repetition penalty works
- [ ] Frequency penalty works
- [ ] Generated text is more diverse

---

## Complete Sample Script

Here's the full `sample.py` with all features:

```python
# sample.py - Complete version
# (Combines all code from Steps 1-7)

# Run with:
#   python sample.py --out_dir=out-shakespeare-char
#   python sample.py --out_dir=out-shakespeare-char --interactive
#   python sample.py --out_dir=out-shakespeare-char --batch
```

---

## Testing Generation

### Test Script

Create `test_generation.py`:

```python
# test_generation.py
"""
Test text generation.
"""

import subprocess
import sys

def test_basic_generation():
    """Test basic generation."""
    print("Testing basic generation...")

    result = subprocess.run([
        sys.executable, 'sample.py',
        '--out_dir=out-shakespeare-char',
        '--num_samples=2',
        '--max_new_tokens=100',
        '--temperature=0.8'
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✓ Basic generation successful")
        print(f"Output length: {len(result.stdout)} characters")
        return True
    else:
        print("✗ Basic generation failed")
        print(result.stderr)
        return False


def test_custom_prompt():
    """Test with custom prompt."""
    print("\nTesting custom prompt...")

    result = subprocess.run([
        sys.executable, 'sample.py',
        '--out_dir=out-shakespeare-char',
        '--start=ROMEO:',
        '--max_new_tokens=50',
        '--num_samples=1'
    ], capture_output=True, text=True)

    if result.returncode == 0:
        output = result.stdout
        if 'ROMEO:' in output:
            print("✓ Custom prompt successful")
            return True
        else:
            print("✗ Custom prompt not in output")
            return False
    else:
        print("✗ Custom prompt failed")
        return False


def test_temperature_variation():
    """Test different temperatures."""
    print("\nTesting temperature variation...")

    temps = [0.1, 0.8, 1.5]
    success = True

    for temp in temps:
        result = subprocess.run([
            sys.executable, 'sample.py',
            '--out_dir=out-shakespeare-char',
            f'--temperature={temp}',
            '--max_new_tokens=50',
            '--num_samples=1'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  ✓ Temperature {temp} works")
        else:
            print(f"  ✗ Temperature {temp} failed")
            success = False

    return success


if __name__ == '__main__':
    print("=" * 60)
    print("Generation Tests")
    print("=" * 60 + "\n")

    results = []
    results.append(test_basic_generation())
    results.append(test_custom_prompt())
    results.append(test_temperature_variation())

    print("\n" + "=" * 60)
    if all(results):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)
```

Run tests:
```bash
python test_generation.py
```

---

## Quality Evaluation

### Metrics to Check

1. **Coherence**: Does the text make sense?
2. **Style matching**: Does it match training data style?
3. **Diversity**: Are samples different from each other?
4. **Grammar**: Is the grammar correct?
5. **Repetition**: Does it get stuck in loops?

### Manual Evaluation

Generate several samples and check:
```bash
python sample.py --out_dir=out-shakespeare-char --num_samples=10 --max_new_tokens=300
```

Look for:
- ✓ Grammatically correct sentences
- ✓ Coherent dialogue/narrative
- ✓ Appropriate style (Shakespearean)
- ✗ Repetitive phrases
- ✗ Nonsensical words
- ✗ Getting stuck in loops

### Example Quality Comparison

**Good generation (temp=0.8, top_k=200):**
```
ROMEO:
What light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief.
```

**Too deterministic (temp=0.1):**
```
ROMEO:
I love you.
I love you.
I love you very much.
I love you very much indeed.
```

**Too creative (temp=2.0, no top_k):**
```
ROMEO:
Zqxwvutsrqponmlkjihgfedcba wuggle
flarbnox zum praktik ventoozle...
```

---

## Troubleshooting

### Issue: Generated text is gibberish

**Causes**:
- Model not trained enough
- Wrong checkpoint loaded
- Temperature too high

**Solutions**:
1. Train longer
2. Verify checkpoint path
3. Lower temperature (0.7-0.9)
4. Use top_k (100-200)

### Issue: Text is repetitive

**Causes**:
- Temperature too low
- Top-k too small
- Model overfitting

**Solutions**:
1. Increase temperature (0.9-1.1)
2. Increase top_k (200-500)
3. Use repetition penalty (1.1-1.3)
4. Train on more diverse data

### Issue: Generation is slow

**Solutions**:
1. Enable compilation: `--compile=True`
2. Use bfloat16 or float16
3. Reduce max_new_tokens
4. Use smaller model
5. Batch multiple prompts together

### Issue: Out of memory during generation

**Solutions**:
1. Reduce max_new_tokens
2. Use smaller model
3. Generate one sample at a time (not batch)
4. Move to CPU (slower but more memory)

---

## Phase 5 Complete!

### What You've Built

✅ **Complete Generation System**:
- Checkpoint loading for inference
- Autoregressive text generation
- Temperature-based sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Interactive generation mode
- Batch generation
- Repetition and frequency penalties
- Configurable sampling parameters

### Key Files Created

- `sample.py` (~400 lines): Complete generation script
- `test_generation.py`: Generation tests

### Generation Capabilities

**Sampling Strategies:**
- Temperature: Control randomness
- Top-k: Filter unlikely tokens
- Top-p: Nucleus sampling
- Penalties: Reduce repetition

**Generation Modes:**
- Batch: Generate multiple samples
- Interactive: Chat-like interface
- File: Load prompts from files

### Example Generations

**Shakespeare (after 5000 iterations):**
```
ROMEO:
What light through yonder window breaks?
It is the east, and Juliet is the sun.
```

**Fine-tuned GPT-2 XL:**
```
HAMLET:
To be, or not to be--that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune...
```

### Next Steps

**Proceed to Phase 6: Advanced Optimizations**
- PyTorch 2.0 compilation
- Flash Attention integration
- Gradient checkpointing
- Multi-node distributed training
- Performance profiling
- Benchmarking utilities

**Estimated time for Phase 6**: 3-5 hours

---

**Phase 5 Character Count**: ~27,800 characters
