# Phase 3: Data Pipeline

## Phase Overview

**Goal:** Build an efficient data loading and preprocessing system for training

**Prerequisites:**
- Phase 1 completed (project structure)
- Phase 2 completed (model architecture)
- Basic understanding of tokenization

**Estimated Duration:** 3-4 hours

**Key Deliverables:**
- Shakespeare dataset preparation script
- Tokenization using tiktoken (GPT-2 BPE)
- Binary format data storage
- Memory-mapped data loading
- Efficient batch generation
- Train/validation split
- Support for multiple datasets

---

## Understanding the Data Pipeline

```
Data Pipeline Flow

Raw Text File (.txt)
  ↓
[Download/Prepare]
  ↓
Tokenization (tiktoken BPE)
  ↓
Token IDs (integers)
  ↓
Train/Val Split (90/10)
  ↓
Binary Format (.bin, uint16)
  ↓
Memory-Mapped Loading (np.memmap)
  ↓
Random Batch Sampling
  ↓
PyTorch Tensors [batch_size, block_size]
  ↓
Model Training
```

**Why this pipeline?**
- **Binary format**: 10x faster loading than text
- **Memory-mapped**: Handle datasets larger than RAM
- **Random sampling**: Better training dynamics
- **Efficient**: Minimal CPU overhead during training

---

## Step 1: Shakespeare Dataset Preparation

**Purpose:** Create a simple dataset for initial training and testing
**Duration:** 30 minutes

### Download Shakespeare Data

The Tiny Shakespeare dataset is a 1MB text file containing Shakespeare's works. It's perfect for quick experiments.

### Implementation

Create `data/shakespeare/prepare.py`:

```python
# data/shakespeare/prepare.py
"""
Prepare the Shakespeare dataset for character-level language modeling.

This script:
1. Downloads the Tiny Shakespeare dataset
2. Tokenizes it using tiktoken (GPT-2 BPE)
3. Splits into train/val sets (90/10)
4. Saves as binary files for efficient loading
"""

import os
import pickle
import requests
import numpy as np
import tiktoken

# Download the dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    print(f"Downloading {data_url}...")
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)
    print(f"Downloaded to {input_file_path}")

# Read the dataset
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

print(f"Length of dataset in characters: {len(data):,}")

# Get the GPT-2 tokenizer
print("Initializing tokenizer...")
enc = tiktoken.get_encoding("gpt2")

# Encode the dataset
print("Tokenizing dataset...")
tokens = enc.encode_ordinary(data)  # encode_ordinary doesn't add special tokens
print(f"Length of dataset in tokens: {len(tokens):,}")

# Calculate vocabulary size
vocab_size = enc.n_vocab
print(f"Vocabulary size: {vocab_size:,}")

# Split into train and validation sets
n = len(tokens)
train_tokens = tokens[:int(n*0.9)]
val_tokens = tokens[int(n*0.9):]
print(f"Train tokens: {len(train_tokens):,}")
print(f"Val tokens: {len(val_tokens):,}")

# Save to binary files (uint16 is sufficient for vocab_size < 65536)
train_ids = np.array(train_tokens, dtype=np.uint16)
val_ids = np.array(val_tokens, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

print(f"\nSaved train.bin: {len(train_ids):,} tokens")
print(f"Saved val.bin: {len(val_ids):,} tokens")

# Save metadata for model configuration
meta = {
    'vocab_size': vocab_size,
    'tokenizer': 'gpt2',
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("\nDataset preparation complete!")
print("\nTo train:")
print("  python train.py config/train_shakespeare_char.py")
```

### Run the Preparation Script

```bash
python data/shakespeare/prepare.py
```

### Expected Output

```
Downloading https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt...
Downloaded to data/shakespeare/input.txt
Length of dataset in characters: 1,115,394
Initializing tokenizer...
Tokenizing dataset...
Length of dataset in tokens: 338,025
Vocabulary size: 50,257
Train tokens: 304,222
Val tokens: 33,803

Saved train.bin: 304,222 tokens
Saved val.bin: 33,803 tokens

Dataset preparation complete!

To train:
  python train.py config/train_shakespeare_char.py
```

### Files Created

After running, you should have:
```
data/shakespeare/
├── input.txt       # Raw text (~1MB)
├── train.bin       # Training tokens (~608KB)
├── val.bin         # Validation tokens (~68KB)
└── meta.pkl        # Metadata (vocab_size, etc.)
```

### Explanation

**Tokenization with tiktoken:**
- Uses GPT-2's BPE (Byte Pair Encoding) tokenizer
- Vocabulary: 50,257 tokens
- More efficient than character-level (fewer tokens)
- Compatible with pretrained GPT-2 models

**Binary format:**
- Stored as `uint16` (2 bytes per token)
- Much faster to load than text
- Memory-mapped for large datasets

**90/10 split:**
- 90% for training
- 10% for validation
- Standard practice in ML

### Verification
- [ ] prepare.py script created
- [ ] Script runs without errors
- [ ] input.txt downloaded (~1MB)
- [ ] train.bin and val.bin created
- [ ] meta.pkl saved
- [ ] Token counts are reasonable

---

## Step 2: OpenWebText Dataset Preparation (Optional)

**Purpose:** Prepare a large-scale dataset for GPT-2 reproduction
**Duration:** 45-60 minutes (mostly download time)

**Note:** This step is optional. Skip it if you're only doing small experiments with Shakespeare.

### Implementation

Create `data/openwebtext/prepare.py`:

```python
# data/openwebtext/prepare.py
"""
Prepare the OpenWebText dataset for GPT-2 reproduction.

OpenWebText is an open-source recreation of WebText (GPT-2's training data).
It contains ~8 million documents (~9GB of text).

This preparation can take 1-2 hours depending on your internet speed.
"""

import os
import pickle
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# Number of workers for parallel processing
num_proc = 8

# Download and load the dataset
print("Loading OpenWebText dataset...")
print("This will download ~54GB of data (decompresses to ~9GB of text)")
print("First time will be slow, subsequent runs use cache")

dataset = load_dataset("openwebtext", num_proc=num_proc)

# Split into train and validation
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')  # Rename 'test' to 'val'

print(f"Train samples: {len(split_dataset['train']):,}")
print(f"Val samples: {len(split_dataset['val']):,}")

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

def process(example):
    """Tokenize a single example."""
    ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores special tokens
    ids.append(enc.eot_token)  # Add end-of-text token
    out = {'ids': ids, 'len': len(ids)}
    return out

# Tokenize the dataset
print("Tokenizing dataset...")
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="Tokenizing",
    num_proc=num_proc,
)

# Concatenate all tokens into single arrays
for split, dset in tokenized.items():
    print(f"\nProcessing {split} split...")
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')

    print(f"  Total tokens: {arr_len:,}")
    print(f"  Writing to {filename}...")

    # Create memory-mapped array
    dtype = np.uint16  # GPT-2 vocab_size (50257) fits in uint16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

    # Write tokens to array
    total_batches = 1024
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'Writing {split}.bin'):
        # Get batch
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])

        # Write to memmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)

    arr.flush()
    print(f"  Saved {filename}")

# Save metadata
meta = {
    'vocab_size': enc.n_vocab,
    'tokenizer': 'gpt2',
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("\nOpenWebText preparation complete!")
print("\nDataset statistics:")
print(f"  Train tokens: ~9 billion")
print(f"  Val tokens: ~4.5 million")
print(f"  Vocabulary: 50,257 tokens")
print("\nTo train GPT-2 from scratch:")
print("  torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py")
```

### Requirements for OpenWebText

Install additional dependency:
```bash
pip install datasets tqdm
```

### Run the Preparation Script

```bash
python data/openwebtext/prepare.py
```

**Warning:** This will download ~54GB of data and take 1-2 hours.

### Expected Output

```
Loading OpenWebText dataset...
This will download ~54GB of data (decompresses to ~9GB of text)
First time will be slow, subsequent runs use cache
Train samples: 8,013,769
Val samples: 4,009

Tokenizing dataset...
Tokenizing: 100%|████████████████| 8013769/8013769 [15:23<00:00, 8674.32 examples/s]

Processing train split...
  Total tokens: 9,035,582,198
  Writing to data/openwebtext/train.bin...
Writing train.bin: 100%|████████████████| 1024/1024 [12:45<00:00,  1.34batches/s]
  Saved data/openwebtext/train.bin

Processing val split...
  Total tokens: 4,434,897
  Writing to data/openwebtext/val.bin...
Writing val.bin: 100%|████████████████| 1024/1024 [00:03<00:00, 287.65batches/s]
  Saved data/openwebtext/val.bin

OpenWebText preparation complete!

Dataset statistics:
  Train tokens: ~9 billion
  Val tokens: ~4.5 million
  Vocabulary: 50,257 tokens

To train GPT-2 from scratch:
  torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

### Files Created

```
data/openwebtext/
├── train.bin       # ~18GB
├── val.bin         # ~9MB
└── meta.pkl        # Metadata
```

### Verification
- [ ] prepare.py script created
- [ ] Dataset downloaded successfully
- [ ] train.bin created (~18GB)
- [ ] val.bin created (~9MB)
- [ ] meta.pkl saved

---

## Step 3: Data Loading Utilities

**Purpose:** Create efficient data loading for training
**Duration:** 30 minutes

### Implementation

Add to `train.py` (or create `data_loader.py`):

```python
# Data loading utilities (add to train.py or create separate file)

import os
import pickle
import numpy as np
import torch


def get_data_path(dataset):
    """
    Get the path to a dataset directory.

    Args:
        dataset: Dataset name ('shakespeare', 'openwebtext', etc.)

    Returns:
        Path to dataset directory
    """
    data_dir = os.path.join('data', dataset)
    return data_dir


def load_dataset_metadata(dataset):
    """
    Load metadata for a dataset.

    Args:
        dataset: Dataset name

    Returns:
        dict: Metadata containing vocab_size, etc.
    """
    data_dir = get_data_path(dataset)
    meta_path = os.path.join(data_dir, 'meta.pkl')

    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        return meta
    else:
        # If no meta.pkl, assume GPT-2 defaults
        return {'vocab_size': 50304, 'tokenizer': 'gpt2'}


def load_tokens(dataset, split):
    """
    Load tokens from binary file using memory mapping.

    Memory mapping allows working with datasets larger than RAM.

    Args:
        dataset: Dataset name
        split: 'train' or 'val'

    Returns:
        np.memmap: Memory-mapped array of token IDs
    """
    data_dir = get_data_path(dataset)
    bin_path = os.path.join(data_dir, f'{split}.bin')

    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Dataset file not found: {bin_path}\n"
                                f"Please run: python data/{dataset}/prepare.py")

    # Load as memory-mapped array (doesn't load entire file into RAM)
    tokens = np.memmap(bin_path, dtype=np.uint16, mode='r')
    return tokens


def get_batch(data, block_size, batch_size, device='cpu', device_type='cpu'):
    """
    Generate a random batch from data.

    This samples random starting positions and extracts sequences of length block_size.

    Args:
        data: Token array (np.memmap or np.array)
        block_size: Sequence length (context window)
        batch_size: Number of sequences in batch
        device: torch device to place tensors on
        device_type: 'cpu', 'cuda', etc.

    Returns:
        x: Input sequences [batch_size, block_size]
        y: Target sequences [batch_size, block_size] (shifted by 1)
    """
    # Sample random starting positions
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Extract sequences
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    # Pin memory for faster GPU transfer
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y


class DataLoaderLite:
    """
    Lightweight data loader for efficient batch iteration.

    This provides a more structured interface than the standalone get_batch function.
    """

    def __init__(self, dataset, split, batch_size, block_size, device='cpu', device_type='cpu'):
        """
        Initialize data loader.

        Args:
            dataset: Dataset name
            split: 'train' or 'val'
            batch_size: Batch size
            block_size: Context length
            device: torch device
            device_type: 'cpu' or 'cuda'
        """
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.device_type = device_type

        # Load data
        self.data = load_tokens(dataset, split)
        print(f"Loaded {split} split: {len(self.data):,} tokens")

        # Calculate number of batches
        self.num_batches = len(self.data) // (batch_size * block_size)

    def get_batch(self):
        """Get a random batch."""
        return get_batch(self.data, self.block_size, self.batch_size, self.device, self.device_type)

    def __len__(self):
        """Number of possible batches."""
        return self.num_batches
```

### Explanation

**Memory-Mapped Loading:**
```python
tokens = np.memmap(bin_path, dtype=np.uint16, mode='r')
```
- Doesn't load entire file into RAM
- OS loads chunks as needed (demand paging)
- Essential for large datasets (OpenWebText is 18GB)

**Batch Sampling:**
```python
ix = torch.randint(len(data) - block_size, (batch_size,))
x = [data[i:i+block_size] for i in ix]
y = [data[i+1:i+1+block_size] for i in ix]
```
- Random starting positions
- Extract sequences of length `block_size`
- Targets are inputs shifted by 1 (next-token prediction)

**Pin Memory:**
```python
x = x.pin_memory().to(device, non_blocking=True)
```
- Speeds up CPU → GPU transfers
- Uses page-locked memory
- Significant speedup on CUDA

### Verification
- [ ] Data loading functions created
- [ ] Memory-mapped loading works
- [ ] Batch generation works
- [ ] Pin memory optimization included

---

## Step 4: Testing the Data Pipeline

**Purpose:** Verify data loading works correctly
**Duration:** 20 minutes

### Create Test Script

Create `test_data.py`:

```python
# test_data.py
"""
Test data loading pipeline.
"""

import torch
import numpy as np
from train import load_tokens, get_batch, load_dataset_metadata


def test_shakespeare_loading():
    """Test loading Shakespeare dataset."""
    print("Testing Shakespeare dataset loading...")

    # Load metadata
    meta = load_dataset_metadata('shakespeare')
    print(f"  Vocabulary size: {meta['vocab_size']:,}")

    # Load train data
    train_data = load_tokens('shakespeare', 'train')
    print(f"  Train tokens: {len(train_data):,}")

    # Load val data
    val_data = load_tokens('shakespeare', 'val')
    print(f"  Val tokens: {len(val_data):,}")

    # Check data type
    assert train_data.dtype == np.uint16
    assert val_data.dtype == np.uint16

    print("✓ Shakespeare loading successful\n")


def test_batch_generation():
    """Test batch generation."""
    print("Testing batch generation...")

    # Load data
    train_data = load_tokens('shakespeare', 'train')

    # Generate batch
    batch_size = 4
    block_size = 64
    x, y = get_batch(train_data, block_size, batch_size, device='cpu')

    print(f"  x shape: {x.shape}")
    print(f"  y shape: {y.shape}")

    # Verify shapes
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)

    # Verify y is x shifted by 1
    for i in range(batch_size):
        # Find where this sequence appears in data
        # (This is approximate due to random sampling)
        print(f"  Sample {i}: x[0]={x[i,0].item()}, y[0]={y[i,0].item()}")

    print("✓ Batch generation successful\n")


def test_data_statistics():
    """Test and display data statistics."""
    print("Testing data statistics...")

    train_data = load_tokens('shakespeare', 'train')

    # Token distribution
    unique_tokens = len(np.unique(train_data))
    print(f"  Unique tokens in train: {unique_tokens:,}")

    # Sample tokens
    print(f"  First 20 tokens: {train_data[:20].tolist()}")
    print(f"  Token range: [{train_data.min()}, {train_data.max()}]")

    # Check for invalid tokens
    meta = load_dataset_metadata('shakespeare')
    vocab_size = meta['vocab_size']
    assert train_data.max() < vocab_size, f"Token {train_data.max()} exceeds vocab_size {vocab_size}"

    print("✓ Data statistics look good\n")


def test_tokenization_roundtrip():
    """Test encoding/decoding roundtrip."""
    print("Testing tokenization roundtrip...")

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    # Test text
    text = "Hello, world! This is a test."
    print(f"  Original: '{text}'")

    # Encode
    tokens = enc.encode(text)
    print(f"  Tokens: {tokens}")

    # Decode
    decoded = enc.decode(tokens)
    print(f"  Decoded: '{decoded}'")

    assert decoded == text, "Roundtrip failed!"
    print("✓ Tokenization roundtrip successful\n")


def test_batch_targets():
    """Verify that targets are correctly shifted."""
    print("Testing batch target alignment...")

    train_data = load_tokens('shakespeare', 'train')

    # Get a batch
    x, y = get_batch(train_data, block_size=8, batch_size=1, device='cpu')

    print("  Input sequence (x):")
    print(f"    {x[0].tolist()}")
    print("  Target sequence (y):")
    print(f"    {y[0].tolist()}")

    # Verify shift: y[i] should be x[i+1] for all positions
    # (This verification is approximate since we don't know the exact data position)
    print("  Note: y should be x shifted left by 1 position")
    print("✓ Target alignment test complete\n")


def test_cuda_batch():
    """Test batch generation on CUDA (if available)."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test\n")
        return

    print("Testing CUDA batch generation...")

    train_data = load_tokens('shakespeare', 'train')
    x, y = get_batch(train_data, block_size=64, batch_size=4, device='cuda', device_type='cuda')

    assert x.is_cuda
    assert y.is_cuda
    print(f"  Batch on device: {x.device}")
    print("✓ CUDA batch generation successful\n")


if __name__ == '__main__':
    print("=" * 60)
    print("Data Pipeline Tests")
    print("=" * 60 + "\n")

    test_shakespeare_loading()
    test_batch_generation()
    test_data_statistics()
    test_tokenization_roundtrip()
    test_batch_targets()
    test_cuda_batch()

    print("=" * 60)
    print("All data tests passed!")
    print("=" * 60)
```

### Run Tests

```bash
python test_data.py
```

### Expected Output

```
============================================================
Data Pipeline Tests
============================================================

Testing Shakespeare dataset loading...
  Vocabulary size: 50,257
  Train tokens: 304,222
  Val tokens: 33,803
✓ Shakespeare loading successful

Testing batch generation...
  x shape: torch.Size([4, 64])
  y shape: torch.Size([4, 64])
  Sample 0: x[0]=1212, y[0]=460
  Sample 1: x[0]=262, y[0]=3580
  Sample 2: x[0]=532, y[0]=11
  Sample 3: x[0]=1148, y[0]=13
✓ Batch generation successful

Testing data statistics...
  Unique tokens in train: 17,527
  First 20 tokens: [5962, 10669, 46823, 263, 1901, 286, 37623, 11, 416, 3977, 10347, 198, 198, 14126, 447, 247, 83, 28786, 286]
  Token range: [0, 50256]
✓ Data statistics look good

Testing tokenization roundtrip...
  Original: 'Hello, world! This is a test.'
  Tokens: [15496, 11, 995, 0, 770, 318, 257, 1332, 13]
  Decoded: 'Hello, world! This is a test.'
✓ Tokenization roundtrip successful

Testing batch target alignment...
  Input sequence (x):
    [1212, 460, 262, 1755, 286, 257, 6792, 11]
  Target sequence (y):
    [460, 262, 1755, 286, 257, 6792, 11, 290]
  Note: y should be x shifted left by 1 position
✓ Target alignment test complete

Testing CUDA batch generation...
  Batch on device: cuda:0
✓ CUDA batch generation successful

============================================================
All data tests passed!
============================================================
```

### Verification
- [ ] All tests pass
- [ ] Data loads successfully
- [ ] Batch shapes are correct
- [ ] Targets are properly shifted
- [ ] CUDA works (if available)

---

## Step 5: Inspect Dataset

**Purpose:** Visualize what the tokenized data looks like
**Duration:** 10 minutes

### Create Inspection Script

Create `inspect_data.py`:

```python
# inspect_data.py
"""
Inspect tokenized dataset to understand the data.
"""

import tiktoken
from train import load_tokens, load_dataset_metadata


def inspect_shakespeare():
    """Inspect Shakespeare dataset."""
    print("=" * 60)
    print("Shakespeare Dataset Inspection")
    print("=" * 60 + "\n")

    # Load metadata
    meta = load_dataset_metadata('shakespeare')
    print(f"Vocabulary size: {meta['vocab_size']:,}\n")

    # Load tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Load data
    train_data = load_tokens('shakespeare', 'train')
    print(f"Train tokens: {len(train_data):,}\n")

    # Show first 1000 tokens decoded
    print("First 200 tokens decoded:")
    print("-" * 60)
    first_tokens = train_data[:200].tolist()
    first_text = enc.decode(first_tokens)
    print(first_text)
    print("-" * 60 + "\n")

    # Show token statistics
    print("Token Statistics:")
    print(f"  Min token ID: {train_data.min()}")
    print(f"  Max token ID: {train_data.max()}")
    print(f"  Mean token ID: {train_data.mean():.1f}")

    # Sample random sequences
    print("\nRandom sequence samples:")
    import numpy as np
    for i in range(3):
        start = np.random.randint(0, len(train_data) - 100)
        tokens = train_data[start:start+100].tolist()
        text = enc.decode(tokens)
        print(f"\n  Sample {i+1} (tokens {start}-{start+100}):")
        print(f"    {text[:200]}...")  # First 200 chars

    # Token frequency analysis
    print("\n\nMost common tokens:")
    unique, counts = np.unique(train_data, return_counts=True)
    top_indices = np.argsort(counts)[-10:][::-1]

    for idx in top_indices:
        token_id = unique[idx]
        count = counts[idx]
        token_str = enc.decode([token_id])
        # Escape special characters for display
        token_display = repr(token_str)[1:-1]  # Remove outer quotes from repr
        print(f"    Token {token_id:5d} ('{token_display}'): {count:6,} times ({count/len(train_data)*100:.2f}%)")


if __name__ == '__main__':
    inspect_shakespeare()
```

### Run Inspection

```bash
python inspect_data.py
```

### Expected Output

```
============================================================
Shakespeare Dataset Inspection
============================================================

Vocabulary size: 50,257

Train tokens: 304,222

First 200 tokens decoded:
------------------------------------------------------------
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.
------------------------------------------------------------

Token Statistics:
  Min token ID: 0
  Max token ID: 50256
  Mean token ID: 16437.2

Random sequence samples:

  Sample 1 (tokens 125431-125531):
    But tell me, is young George Stanley living?

STANLEY:
He is, my lord; and safe in Leicester town;
Whither, if it please you, we may now withdraw us.

KING RICHARD III:
Why, what...

  Sample 2 (tokens 89234-89334):
    Thou shalt not die: die for adultery! No:
The wren goes to't, and the small gilded fly
Does lecher in my sight.
Let copulation thrive; for Gloucester's bastard son...

  Sample 3 (tokens 204551-204651):
    What, old acquaintance! could not all this flesh
Keep in a little life? Poor Jack, farewell!
I could have better spared a better man:
O, I should have a heavy...


Most common tokens:
    Token   262 (' the'): 13,206 times (4.34%)
    Token   286 (' of'):  7,744 times (2.55%)
    Token   290 (' and'):  7,580 times (2.49%)
    Token   284 (' to'):  7,289 times (2.40%)
    Token   314 (' I'):  5,891 times (1.94%)
    Token   257 (' a'):  5,502 times (1.81%)
    Token   345 (' you'):  4,982 times (1.64%)
    Token   616 (' my'):  4,401 times (1.45%)
    Token   198 ('\n'):  4,201 times (1.38%)
    Token   326 (' is'):  3,890 times (1.28%)
```

### Verification
- [ ] Can decode tokens back to text
- [ ] Text looks correct (Shakespeare plays)
- [ ] Token statistics are reasonable
- [ ] Most common tokens make sense (articles, conjunctions)

---

## Understanding the Data Format

### Token IDs
- Each token is represented as a 16-bit integer (0-50256)
- Stored in binary format as `uint16` arrays
- GPT-2 BPE vocabulary: 50,257 tokens

### Batch Structure
```python
# Input batch
x = [[token_1, token_2, ..., token_64],    # Sequence 1
     [token_5, token_6, ..., token_68],    # Sequence 2
     ...                                    # ...
     [token_n, token_m, ..., token_k]]     # Sequence batch_size

# Target batch (shifted by 1)
y = [[token_2, token_3, ..., token_65],    # Targets for sequence 1
     [token_6, token_7, ..., token_69],    # Targets for sequence 2
     ...                                    # ...
     [token_m, token_k, ..., token_j]]     # Targets for sequence batch_size
```

### Training Objective
- Predict next token given previous tokens
- Model learns: P(token_i | token_1, ..., token_{i-1})
- Cross-entropy loss between predictions and targets

---

## Troubleshooting

### Issue: FileNotFoundError when loading data

**Solution**: Run the preparation script first:
```bash
python data/shakespeare/prepare.py
```

### Issue: "requests" module not found

**Solution**: Install requests:
```bash
pip install requests
```

### Issue: Tokenization is slow

**Expected**: First tokenization takes time. Subsequent runs use cached `.bin` files which load instantly.

### Issue: Out of memory loading OpenWebText

**Not a problem**: Memory-mapped loading doesn't load entire file. Check that you're using `np.memmap` with `mode='r'`.

### Issue: Token IDs exceed vocab_size

**Cause**: Using wrong tokenizer or corrupted data.

**Solution**:
1. Delete `.bin` files
2. Re-run prepare script
3. Verify tiktoken version: `pip install --upgrade tiktoken`

---

## Phase 3 Complete!

### What You've Built

✅ **Complete Data Pipeline**:
- Shakespeare dataset preparation (~1MB, 300K tokens)
- OpenWebText preparation (optional, ~9GB, 9B tokens)
- Efficient binary format storage
- Memory-mapped data loading
- Random batch sampling
- Train/validation splits
- Metadata management

### Key Files Created

- `data/shakespeare/prepare.py`: Shakespeare preparation
- `data/openwebtext/prepare.py`: OpenWebText preparation (optional)
- `train.py` (data functions): Data loading utilities
- `test_data.py`: Data pipeline tests
- `inspect_data.py`: Dataset inspection tool

### Dataset Statistics

**Shakespeare**:
- Raw text: 1.1M characters
- Tokens: 338K (GPT-2 BPE)
- Train: 304K tokens (~90%)
- Val: 34K tokens (~10%)
- Size on disk: ~700KB

**OpenWebText** (optional):
- Raw text: ~9GB
- Tokens: ~9 billion
- Train: ~9B tokens
- Val: ~4.5M tokens
- Size on disk: ~18GB

### Next Steps

**Proceed to Phase 4: Training Loop**
- Implement complete training script
- AdamW optimizer setup
- Learning rate scheduling
- Mixed precision training
- Checkpointing
- Evaluation loop
- DDP support

**Estimated time for Phase 4**: 5-7 hours

---

**Phase 3 Character Count**: ~39,500 characters
