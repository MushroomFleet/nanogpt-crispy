# Phase 6C: Distributed Training & Profiling

## Substage Overview

- **Parent Phase:** Phase 6: Advanced Optimizations
- **Substage:** C of 3 (Distributed & Profiling)
- **Goal:** Add multi-node training and performance profiling capabilities
- **Prerequisites:** Phase 6A & 6B completed (speed + memory optimizations)
- **Estimated Duration:** 2 hours
- **Key Deliverables:**
  - Multi-node distributed training setup
  - SLURM job scripts for HPC clusters
  - Launch scripts for manual multi-node coordination
  - PyTorch profiler integration
  - Performance profiling tools
  - Comprehensive benchmarking suite
  - Bottleneck identification

---

## What We're Building

In 6A and 6B, we optimized single-node training. Now we scale to multiple machines:

1. **Multi-Node DDP** - Scale across machines
2. **SLURM Integration** - HPC cluster support
3. **Performance Profiling** - Find bottlenecks
4. **Comprehensive Benchmarks** - Measure all optimizations

**Note:** Multi-node requires cluster infrastructure (optional for most users)

---

## Previous Substages Recap

**Phase 6A - Speed:**
- ✅ torch.compile (2-4x speedup)
- ✅ Flash Attention (2-3x speed + memory)

**Phase 6B - Memory:**
- ✅ Gradient checkpointing (30-50% memory reduction)

Now we add distributed training and profiling!

---

## Step 1: Multi-Node Distributed Training

**Purpose:** Scale training across multiple machines  
**Duration:** 60 minutes

### Understanding Multi-Node DDP

**Single-Node DDP (Phase 4):**
- Multiple GPUs on one machine
- Fast communication (NVLink/PCIe)
- Started with `torchrun`

**Multi-Node DDP (this step):**
- Multiple machines, each with multiple GPUs
- Communication via network
- Requires coordination between nodes
- Much more complex setup

### Requirements

- Multiple machines with GPUs
- Network connectivity
- Shared filesystem (NFS) or network storage
- Same PyTorch version on all nodes

### Update train.py for Multi-Node

Add to the DDP section in `train.py`:

```python
# Multi-node DDP configuration
# torchrun sets these environment variables
ddp_world_size = int(os.environ.get('WORLD_SIZE', 1))
ddp_rank = int(os.environ.get('RANK', -1))
ddp_local_rank = int(os.environ.get('LOCAL_RANK', -1))

# For multi-node
master_addr = os.environ.get('MASTER_ADDR', 'localhost')
master_port = os.environ.get('MASTER_PORT', '12355')

# Initialize process group for multi-node
if ddp_rank != -1:
    torch.cuda.set_device(ddp_local_rank)
    device = f'cuda:{ddp_local_rank}'

    init_process_group(
        backend=backend,
        init_method=f'tcp://{master_addr}:{master_port}',
        world_size=ddp_world_size,
        rank=ddp_rank
    )

    master_process = (ddp_rank == 0)
    print(f"Rank {ddp_rank}/{ddp_world_size} initialized on {master_addr}")
```

### Create Launch Script

Create `launch_multinode.sh`:

```bash
#!/bin/bash
# launch_multinode.sh
# Launch multi-node distributed training

# Configuration
NUM_NODES=2
GPUS_PER_NODE=8
MASTER_ADDR="node1.cluster"  # Change to your master node address
MASTER_PORT=12355

# Node rank (different for each node)
NODE_RANK=$1  # Pass as argument: ./launch_multinode.sh 0 (for node 0)

if [ -z "$NODE_RANK" ]; then
    echo "Usage: ./launch_multinode.sh <node_rank>"
    echo "  node_rank: 0 for master, 1, 2, ... for workers"
    exit 1
fi

# Calculate world size
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

echo "Launching node rank $NODE_RANK of $NUM_NODES"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  Total GPUs: $WORLD_SIZE"

# Launch training
torchrun \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py config/train_gpt2.py

echo "Node $NODE_RANK finished"
```

### Usage

On each node:

```bash
# Node 0 (master)
./launch_multinode.sh 0

# Node 1 (worker)
./launch_multinode.sh 1
```

---

## Step 2: SLURM Integration

**Purpose:** Support HPC clusters with SLURM  
**Duration:** 20 minutes

### Create submit_slurm.sh

```bash
#!/bin/bash
#SBATCH --job-name=nanogpt
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Load modules (adjust for your cluster)
module load cuda/11.8
module load pytorch/2.0

# Set master address
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12355

# Launch training
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py config/train_gpt2.py
```

### Submit Job

```bash
sbatch submit_slurm.sh
```

### Monitor Job

```bash
# Check status
squeue -u $USER

# View output
tail -f slurm-<job_id>.out
```

---

## Step 3: Performance Profiling

**Purpose:** Identify training bottlenecks  
**Duration:** 30 minutes

### Create profile_training.py

```python
# profile_training.py
"""
Profile training to identify performance bottlenecks.
"""

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from model import GPT, GPTConfig


def profile_step(model, x, y, optimizer):
    """Profile a single training step."""
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(x, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def run_profiling():
    """Run profiling on training."""
    print("Starting profiling...\n")

    # Setup
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Dummy data
    batch_size = 8
    seq_len = 1024
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("training_step"):
            for _ in range(10):
                profile_step(model, x, y, optimizer)

    # Print results
    print("\n" + "=" * 80)
    print("Top 10 operations by CUDA time:")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10
    ))

    print("\n" + "=" * 80)
    print("Top 10 operations by CPU time:")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=10
    ))

    print("\n" + "=" * 80)
    print("Memory usage:")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=10
    ))

    # Export to Chrome trace
    prof.export_chrome_trace("trace.json")
    print("\nProfile exported to trace.json")
    print("View in Chrome: chrome://tracing")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available, skipping profiling")
        exit(0)

    run_profiling()
```

### Run Profiling

```bash
python profile_training.py
```

### Expected Output

```
Starting profiling...

================================================================================
Top 10 operations by CUDA time:
================================================================================
Name                                    Self CPU    Self CUDA   Calls
---------------------------------  ------------  -----------  ------
aten::mm                                 2.45ms      45.23ms     144
aten::addmm                              1.23ms      32.45ms      72
aten::linear                             0.89ms      28.76ms     216
aten::_scaled_dot_product_flash...       0.56ms      24.32ms      12
...

================================================================================
Memory usage:
================================================================================
Name                              Self CPU Mem   Self CUDA Mem
---------------------------------  -----------  -------------
aten::empty                              0 b        2.34 Gb
aten::linear                             0 b        1.23 Gb
...

Profile exported to trace.json
View in Chrome: chrome://tracing
```

### Analyze in Chrome

1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Load `trace.json`
4. Visualize timeline of operations
5. Identify gaps (CPU waiting for GPU, or vice versa)

---

## Step 4: Comprehensive Benchmark Suite

**Purpose:** Compare all optimizations  
**Duration:** 20 minutes

### Create benchmark_all.py

```python
# benchmark_all.py
"""
Comprehensive benchmark of all optimizations.
"""

import time
import torch
from model import GPT, GPTConfig


def run_benchmark(name, model, batch_size=8, seq_len=512, num_iters=50):
    """Run benchmark for a configuration."""
    device = next(model.parameters()).device

    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(5):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Benchmark
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

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

    avg_time = sum(times) / len(times) * 1000
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    return avg_time, peak_mem


if __name__ == '__main__':
    print("=" * 80)
    print("Comprehensive Optimization Benchmark")
    print("=" * 80 + "\n")

    if not torch.cuda.is_available():
        print("CUDA not available")
        exit(0)

    config = GPTConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=1024,
        vocab_size=50304,
        dropout=0.0
    )

    results = []

    # 1. Baseline
    print("1. Baseline (no optimizations)")
    model = GPT(config).to('cuda')
    time_ms, mem_gb = run_benchmark("Baseline", model)
    results.append(("Baseline", time_ms, mem_gb))
    print(f"   Time: {time_ms:.2f}ms, Memory: {mem_gb:.2f}GB\n")

    # 2. Compiled
    print("2. + torch.compile")
    model = GPT(config).to('cuda')
    model = torch.compile(model)
    time_ms, mem_gb = run_benchmark("Compiled", model)
    results.append(("+ Compiled", time_ms, mem_gb))
    print(f"   Time: {time_ms:.2f}ms, Memory: {mem_gb:.2f}GB\n")

    # 3. Gradient checkpointing
    print("3. + Gradient Checkpointing")
    model = GPT(config).to('cuda')
    model.enable_gradient_checkpointing()
    time_ms, mem_gb = run_benchmark("Grad Checkpoint", model)
    results.append(("+ Grad Checkpoint", time_ms, mem_gb))
    print(f"   Time: {time_ms:.2f}ms, Memory: {mem_gb:.2f}GB\n")

    # 4. All optimizations
    print("4. + All (Compile + Flash + Checkpoint)")
    model = GPT(config).to('cuda')
    model.enable_gradient_checkpointing()
    model = torch.compile(model)
    time_ms, mem_gb = run_benchmark("All Optimizations", model)
    results.append(("+ All", time_ms, mem_gb))
    print(f"   Time: {time_ms:.2f}ms, Memory: {mem_gb:.2f}GB\n")

    # Summary
    print("=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"{'Configuration':<30} {'Time (ms)':<15} {'Memory (GB)':<15} {'Speedup':<10}")
    print("-" * 80)

    baseline_time = results[0][1]
    baseline_mem = results[0][2]

    for name, time_ms, mem_gb in results:
        speedup = baseline_time / time_ms
        mem_ratio = mem_gb / baseline_mem
        print(f"{name:<30} {time_ms:>10.2f}     {mem_gb:>10.2f}     {speedup:>8.2f}x")

    print("=" * 80)
```

### Run Benchmark

```bash
python benchmark_all.py
```

### Expected Output

```
================================================================================
Comprehensive Optimization Benchmark
================================================================================

1. Baseline (no optimizations)
   Time: 245.67ms, Memory: 12.34GB

2. + torch.compile
   Time: 115.23ms, Memory: 11.76GB

3. + Gradient Checkpointing
   Time: 298.32ms, Memory: 7.23GB

4. + All (Compile + Flash + Checkpoint)
   Time: 142.45ms, Memory: 6.89GB

================================================================================
Summary:
================================================================================
Configuration                  Time (ms)       Memory (GB)     Speedup
--------------------------------------------------------------------------------
Baseline                          245.67          12.34        1.00x
+ Compiled                        115.23          11.76        2.13x
+ Grad Checkpoint                 298.32           7.23        0.82x
+ All                             142.45           6.89        1.72x
================================================================================
```

**Key insights:**
- Compile gives speed, slight memory improvement
- Checkpointing gives memory, trades speed
- Combined: Better speed than baseline, much less memory

---

## Step 5: Bottleneck Identification

**Purpose:** Find what's slowing down training  
**Duration:** 15 minutes

### Analyzing Profiler Output

When you run `profile_training.py`, look for:

**1. Top CUDA operations:**
- Should be dominated by `mm` (matrix multiply), `addmm`
- If you see `copy` operations: data transfer bottleneck
- If you see `synchronize`: CPU-GPU sync issues

**2. CPU vs CUDA time:**
- CUDA time >> CPU time: Good (GPU utilized)
- CPU time >> CUDA time: Bad (GPU waiting for CPU)

**3. Memory allocations:**
- Large allocations: `aten::empty`, `aten::linear`
- Repeated large allocations: potential optimization target

### Common Bottlenecks

**Data loading bottleneck:**
```
Symptom: High CPU time for data operations
Solution: Use pin_memory, num_workers in DataLoader
```

**CPU-GPU transfer bottleneck:**
```
Symptom: High time in `aten::copy_`
Solution: Pin memory, pre-load batches
```

**Attention bottleneck:**
```
Symptom: High time in attention operations
Solution: Use Flash Attention (Phase 6A)
```

---

## Step 6: Production Scaling Examples

**Purpose:** Real-world scaling scenarios  
**Duration:** 10 minutes

### Scaling Path

**Start: Single GPU**
```bash
python train.py config/train_shakespeare_char.py
# ~10 minutes, 1 GPU
```

**Scale to 4 GPUs:**
```bash
torchrun --standalone --nproc_per_node=4 train.py config/train_shakespeare_char.py
# ~3 minutes, 4 GPUs (~3.5x speedup)
```

**Scale to 2 Nodes × 8 GPUs:**
```bash
# On each node:
./launch_multinode.sh 0  # Node 0
./launch_multinode.sh 1  # Node 1
# ~1 minute, 16 GPUs (~14x speedup)
```

### Expected Speedups

**Perfect scaling:** Time = Base_time / num_GPUs

**Reality:**
- 4 GPUs: ~3.5x (87% efficiency)
- 8 GPUs: ~6.5x (81% efficiency)  
- 16 GPUs: ~13x (81% efficiency)
- 64 GPUs: ~45x (70% efficiency)

**Why not perfect?**
- Communication overhead (gradient synchronization)
- Network latency (multi-node)
- Load imbalance
- Python overhead

---

## Troubleshooting

### Issue: Multi-node hangs at initialization

**Solutions:**
1. Check network connectivity between nodes
2. Verify MASTER_ADDR is reachable from all nodes
3. Check firewall: port 12355 must be open
4. Ensure same PyTorch version on all nodes

### Issue: "NCCL error"

**Solutions:**
1. Check NCCL installation: `python -c "import torch; print(torch.cuda.nccl.version())"`
2. Try different backend: `backend = 'gloo'`
3. Check CUDA version compatibility

### Issue: Profiler crashes

**Solutions:**
1. Reduce profiling iterations
2. Disable memory profiling: `profile_memory=False`
3. Use simpler model for profiling

---

## Phase 6C Complete!

### What You've Built

✅ **Distributed Training & Profiling**:
- Multi-node DDP setup
- SLURM integration for HPC clusters
- Launch scripts for manual coordination
- Performance profiling with PyTorch profiler
- Chrome trace visualization
- Comprehensive benchmarking suite
- Bottleneck identification tools

### Key Files Created

- `launch_multinode.sh`: Multi-node launcher
- `submit_slurm.sh`: SLURM job script
- `profile_training.py`: Performance profiler
- `benchmark_all.py`: Comprehensive benchmarks

### Scaling Capabilities

**You can now:**
1. ✅ Train on single GPU (Phases 4-5)
2. ✅ Train on multiple GPUs (Phase 4C)
3. ✅ Train on multiple nodes (Phase 6C)
4. ✅ Use HPC clusters with SLURM
5. ✅ Profile and optimize performance
6. ✅ Benchmark all optimizations

### Performance Summary

**Single GPU (RTX 4090):**
- Baseline: 1.0x, ~12 GB memory
- + Compile: 2.0x, ~11 GB memory
- + Flash: 1.5x, ~9 GB memory
- + Checkpoint: 0.7x, ~6 GB memory
- **+ All: 1.7x, ~6 GB memory** (best overall)

**Multi-GPU (8× A100):**
- Single GPU: 1.0x
- 8 GPUs (DDP): ~6.5x
- With compile + Flash: ~40x total

**Multi-Node (8 nodes × 8 GPUs):**
- Single GPU: 1.0x
- 64 GPUs: ~45x
- With all optimizations: ~250x+

---

## All Phase 6 Substages Complete!

### What You've Accomplished

**Phase 6A - Speed:**
- torch.compile (2-4x speedup)
- Flash Attention (2-3x speed + memory)

**Phase 6B - Memory:**
- Gradient checkpointing (30-50% memory savings)

**Phase 6C - Distributed:**
- Multi-node training (scale to clusters)
- Performance profiling (find bottlenecks)
- Comprehensive benchmarking

### Optimization Decision Tree

```
Do you have enough GPU memory for your model?
├─ YES: Use compile + Flash Attention (max speed)
└─ NO: Use compile + Flash + Checkpoint (balanced)

Do you have multiple GPUs?
├─ YES: Use DDP (Phase 4C)
└─ NO: Single GPU is fine

Do you have a cluster?
├─ YES: Use multi-node DDP (Phase 6C)
└─ NO: Single node is fine

Need to debug performance?
└─ Use profiler (Phase 6C)
```

---

## Next Steps

**You now have a complete, optimized NanoGPT implementation!**

**Optional enhancements:**
- Add LoRA for efficient fine-tuning
- Implement model quantization (8-bit, 4-bit)
- Add FSDP for even larger models
- Create web demo (Gradio/Streamlit)
- Implement evaluation suite
- Add experiment tracking (W&B)

**Production deployment:**
- Optimize for inference (TorchScript, ONNX)
- Add serving layer (FastAPI)
- Implement caching
- Monitor performance

---

**Phase 6C Character Count:** ~14,900 characters

**All Educational Substages Complete!** 🎉
