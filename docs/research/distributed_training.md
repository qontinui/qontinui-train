# Distributed Training Frameworks and Strategies

**Last Updated:** November 2024

This document provides a comprehensive overview of distributed training frameworks, strategies, and best practices for training large-scale neural networks, with a focus on GUI understanding models.

---

## Table of Contents

1. [Framework Comparison](#framework-comparison)
2. [DeepSpeed ZeRO](#deepspeed-zero)
3. [PyTorch Lightning Integration](#pytorch-lightning-integration)
4. [Optimization Strategies](#optimization-strategies)
5. [Cost and Infrastructure](#cost-and-infrastructure)
6. [Monitoring and Debugging](#monitoring-and-debugging)

---

## 1. Framework Comparison

### Overview of Major Frameworks

#### PyTorch Lightning + DeepSpeed (Recommended)

PyTorch Lightning with DeepSpeed provides a high-level interface for distributed training with advanced memory optimization capabilities.

**Pros:**
- Seamless integration with PyTorch Lightning's trainer abstraction
- Cutting-edge features through DeepSpeed (ZeRO-3 Offload, activation checkpointing)
- Excellent for models with 500M+ parameters
- Strong community support and active development
- Simple configuration through strategy parameter
- Advanced features like ZeRO-3 Offload for single-GPU fine-tuning of 10-20B parameter models

**Cons:**
- Saves checkpoints in proprietary format (can be converted)
- Allocates ~3.6GB VRAM for distributed communications by default
- Steeper learning curve for advanced features
- May have higher communication overhead in some scenarios

**Best For:** Large-scale pre-training and fine-tuning of models with billions of parameters

---

#### Hugging Face Accelerate

Accelerate is a unified interface that simplifies distributed training by wrapping PyTorch FSDP, DeepSpeed, and other frameworks.

**Pros:**
- Unified API for multiple backends (FSDP, DeepSpeed, DDP)
- Minimal code changes required (usually just wrapping model, optimizer, dataloader)
- Seamless integration with Hugging Face ecosystem
- Flexible switching between FSDP and DeepSpeed
- Automatic mixed precision handling
- Support for gradient accumulation and gradient clipping
- Easy configuration via config files or command line

**Cons:**
- Abstraction layer may hide framework-specific optimizations
- Less control over fine-grained configuration
- Not a standalone framework (requires FSDP or DeepSpeed underneath)

**Best For:** Hugging Face Transformers users, rapid prototyping, switching between frameworks

---

#### PyTorch FSDP (Fully Sharded Data Parallel)

Native PyTorch solution for fully sharded data parallel training, now part of core PyTorch.

**Pros:**
- Native PyTorch integration (no external dependencies)
- Production-ready and officially supported by Meta
- FSDP2 introduces DTensor for simpler internal implementation
- Faster checkpointing with asynchronous support in FSDP2
- Works with LoRA and other PEFT methods out-of-the-box (FSDP2)
- Improved precision handling alignment with DeepSpeed (as of Accelerate 0.30.0)
- Enables larger batch sizes and models that fail with standard DDP

**Cons:**
- Feature parity with DeepSpeed not complete
- Less aggressive memory optimization than ZeRO-3
- Documentation less comprehensive than PyTorch Lightning
- Requires more manual configuration

**Best For:** PyTorch-native workflows, production deployments requiring minimal dependencies

---

#### Ray Train

Distributed training framework built on Ray, designed for scalable ML workflows.

**Pros:**
- Unified interface for PyTorch, TensorFlow, XGBoost, LightGBM
- Excellent for multi-framework pipelines
- Strong integration with Ray ecosystem (Tune, Serve, Data)
- Performance parity with native PyTorch Distributed (within 2.5%)
- Scales to 1000+ GPUs (175B parameters trained with 57.5% peak GPU utilization)
- Good for heterogeneous clusters and CPU-based training
- Near-linear scaling demonstrated on terabyte-scale benchmarks

**Cons:**
- Additional abstraction layer
- Ray dependency adds complexity
- Less specialized for pure deep learning compared to Lightning
- Smaller community compared to PyTorch Lightning

**Best For:** Multi-stage ML pipelines, hyperparameter tuning with distributed training, CPU-based training

---

### Framework Comparison Table

| Feature | Lightning + DeepSpeed | Accelerate | PyTorch FSDP | Ray Train |
|---------|----------------------|------------|--------------|-----------|
| **Ease of Use** | High (trainer abstraction) | Very High (minimal changes) | Medium (manual setup) | Medium (Ray concepts) |
| **Memory Efficiency** | Excellent (ZeRO-3) | Excellent (via backends) | Very Good (FSDP2) | Good (depends on backend) |
| **Max Model Size** | 40B+ on single GPU (ZeRO-3 Offload) | 40B+ (via DeepSpeed) | 20B+ (FSDP2) | 175B+ (with Alpa) |
| **Communication Overhead** | Low-Medium | Low-Medium | Low | Low (within 2.5% of native) |
| **Flexibility** | High | Very High | Very High | Very High |
| **Production Ready** | Yes | Yes | Yes | Yes |
| **Checkpoint Format** | Proprietary (.ckpt folder) | Standard | Standard | Standard |
| **Multi-Framework** | No (PyTorch only) | No (PyTorch only) | No (PyTorch only) | Yes |
| **GPU Utilization** | Excellent | Excellent | Excellent | 57.5% (175B params) |
| **Learning Curve** | Medium | Low | Medium-High | Medium-High |
| **Best Use Case** | Large model training | HF Transformers, prototyping | PyTorch-native workflows | Multi-framework pipelines |

---

### When to Use Each Framework

**Use PyTorch Lightning + DeepSpeed when:**
- Training models with 500M+ parameters
- You need ZeRO-3 Offload for single-GPU training of large models
- You want a high-level trainer abstraction
- Pre-training large models from scratch

**Use Hugging Face Accelerate when:**
- Working with Hugging Face Transformers
- You want to easily switch between FSDP and DeepSpeed
- Rapid prototyping and experimentation
- Minimal code changes are desired

**Use PyTorch FSDP when:**
- You prefer native PyTorch solutions
- Production deployment with minimal dependencies
- Using LoRA or other PEFT methods (FSDP2)
- You need full control over distributed training

**Use Ray Train when:**
- Building multi-stage ML pipelines
- Combining distributed training with hyperparameter tuning
- Training across heterogeneous resources
- CPU-based distributed training (e.g., ThirdAI)

---

## 2. DeepSpeed ZeRO

### ZeRO (Zero Redundancy Optimizer) Overview

DeepSpeed's ZeRO family of technologies optimizes memory usage by partitioning model states across GPUs, enabling training of models that wouldn't fit in GPU memory otherwise.

### ZeRO Stages Explained

#### Stage 1: Optimizer State Partitioning

**What it does:**
- Partitions optimizer states (e.g., Adam's 32-bit weights, first and second moment estimates) across GPUs
- Each process updates only its partition of optimizer states

**Memory Savings:**
- **4x memory reduction** compared to standard data parallel training
- Example: For a 1B parameter model, reduces optimizer memory from 12GB to 3GB per GPU

**Communication Overhead:**
- Minimal - only requires all-gather of updated parameters
- More efficient or at parity with DDP due to optimized custom communications

**When to Use:**
- Models 500M - 2B parameters
- When you want better memory efficiency than DDP without communication overhead
- Pre-training with sufficient GPU memory for model parameters

**Example Configuration:**
```python
from lightning.pytorch import Trainer

trainer = Trainer(
    accelerator="gpu",
    devices=8,
    strategy="deepspeed_stage_1",
    precision="bf16-mixed"
)
```

---

#### Stage 2: Optimizer + Gradient Partitioning

**What it does:**
- Partitions both optimizer states AND gradients (reduced 16-bit gradients)
- Each process retains only gradients corresponding to its optimizer state partition

**Memory Savings:**
- **8x memory reduction** compared to standard data parallel training
- Approximately 2x better than Stage 1
- Example: For a 10B parameter model, reduces memory from 160GB to 20GB per GPU

**Communication Overhead:**
- Low - requires reduce-scatter for gradients
- Still comparable to or better than DDP performance

**When to Use (Recommended for Pre-training):**
- Large models (10-20B parameters) with 128+ GPUs
- Pre-training scenarios where you want to scale without performance hit
- Most balanced stage for pre-training workloads

**Example Configuration:**
```python
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DeepSpeedStrategy

# Basic Stage 2
trainer = Trainer(
    accelerator="gpu",
    devices=8,
    strategy="deepspeed_stage_2",
    precision="bf16-mixed"
)

# Advanced Stage 2 with custom config
strategy = DeepSpeedStrategy(
    stage=2,
    offload_optimizer=False,
    allgather_bucket_size=5e8,
    reduce_bucket_size=5e8,
)

trainer = Trainer(
    accelerator="gpu",
    devices=8,
    strategy=strategy,
    precision="bf16-mixed"
)
```

---

#### Stage 3: Optimizer + Gradient + Parameter Partitioning

**What it does:**
- Partitions optimizer states, gradients, AND 16-bit model parameters
- Automatically collects and partitions parameters during forward and backward passes
- Most aggressive memory optimization

**Memory Savings:**
- Memory savings **scale with number of GPUs** (N_gpus reduction factor)
- Example: 100B parameter model on 64 GPUs = ~1.6x memory per GPU vs 100x on single GPU
- Enables training of 40B+ parameter models on single GPU with offloading
- Can train 2T+ parameter models on 512 GPUs

**Communication Overhead:**
- Highest among all stages
- Requires all-gather for parameters during forward/backward passes
- May impact throughput for smaller models or slower interconnects

**When to Use (Recommended for Fine-tuning):**
- Fine-tuning 10-20B parameter models on limited GPUs
- Single-GPU training of billion-parameter models (with ZeRO-3 Offload)
- Maximum memory efficiency is required
- Slower training is acceptable for memory gains

**Example Configuration:**
```python
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DeepSpeedStrategy

# Basic Stage 3
trainer = Trainer(
    accelerator="gpu",
    devices=8,
    strategy="deepspeed_stage_3",
    precision="bf16-mixed"
)

# Stage 3 with offloading (for single GPU or extreme memory constraints)
strategy = DeepSpeedStrategy(
    stage=3,
    offload_optimizer=True,  # Offload optimizer to CPU
    offload_parameters=True,  # Offload parameters to CPU
    cpu_checkpointing=True,   # Checkpoint on CPU
)

trainer = Trainer(
    accelerator="gpu",
    devices=1,
    strategy=strategy,
    precision="bf16-mixed"
)
```

---

### ZeRO Stage Selection Guide

| Model Size | GPUs Available | Recommended Stage | Rationale |
|------------|----------------|-------------------|-----------|
| < 500M | Any | Standard DDP | ZeRO overhead not worth it |
| 500M - 2B | 1-8 | ZeRO-1 | Good balance of memory and speed |
| 2B - 10B | 8-32 | ZeRO-2 | Optimal for pre-training |
| 10B - 20B | 64-128 | ZeRO-2 | Scales well with many GPUs |
| 10B - 20B | 1-8 | ZeRO-3 | Necessary for limited GPUs |
| 20B - 100B | 1-4 | ZeRO-3 + Offload | Only option for few GPUs |
| 20B - 100B | 128+ | ZeRO-2 or ZeRO-3 | ZeRO-2 if memory permits |
| 100B+ | Any | ZeRO-3 | Required for extreme scale |

**Pre-training:** Start with ZeRO-2 and scale number of GPUs
**Fine-tuning:** Use ZeRO-3 or ZeRO-3 Offload for maximum memory efficiency

---

### Advanced ZeRO Configuration

#### Complete DeepSpeed Config Example

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,

  "fp16": {
    "enabled": false
  },

  "bf16": {
    "enabled": true
  },

  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "overlap_comm": true,
    "contiguous_gradients": true
  },

  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
  },

  "steps_per_print": 100,
  "wall_clock_breakdown": false
}
```

#### Using Config with PyTorch Lightning

```python
from lightning.pytorch.strategies import DeepSpeedStrategy

strategy = DeepSpeedStrategy(
    config="/path/to/deepspeed_config.json"
)

trainer = Trainer(
    accelerator="gpu",
    devices=8,
    strategy=strategy,
)
```

---

## 3. PyTorch Lightning Integration

### Basic Multi-GPU Setup

```python
from lightning.pytorch import Trainer, LightningModule
import torch
from torch import nn

class VisionModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            # ... more layers
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

# Single-GPU training
trainer = Trainer(
    accelerator="gpu",
    devices=1,
)

# Multi-GPU DDP (for models < 500M params)
trainer = Trainer(
    accelerator="gpu",
    devices=8,  # Use all 8 GPUs
    strategy="ddp",
)

# Multi-node training
trainer = Trainer(
    accelerator="gpu",
    devices=8,
    num_nodes=4,  # 4 nodes × 8 GPUs = 32 GPUs
    strategy="ddp",
)
```

---

### DeepSpeed Strategy Setup

```python
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DeepSpeedStrategy

# Method 1: String-based configuration
trainer = Trainer(
    accelerator="gpu",
    devices=8,
    strategy="deepspeed_stage_2",
    precision="bf16-mixed",
)

# Method 2: DeepSpeedStrategy with custom settings
strategy = DeepSpeedStrategy(
    stage=2,

    # Offloading options
    offload_optimizer=False,
    offload_parameters=False,

    # Communication optimization
    allgather_bucket_size=5e8,
    reduce_bucket_size=5e8,

    # Memory optimization
    contiguous_gradients=True,
    overlap_comm=True,

    # CPU offloading for extreme memory constraints
    cpu_checkpointing=False,
)

trainer = Trainer(
    accelerator="gpu",
    devices=8,
    strategy=strategy,
    precision="bf16-mixed",
)

# Method 3: Using JSON config file
strategy = DeepSpeedStrategy(
    config="/path/to/deepspeed_config.json"
)

trainer = Trainer(
    accelerator="gpu",
    devices=8,
    strategy=strategy,
)
```

---

### Mixed Precision Training (FP16/BF16)

```python
from lightning.pytorch import Trainer

# BF16 Mixed Precision (Recommended for A100/H100)
trainer = Trainer(
    accelerator="gpu",
    devices=8,
    precision="bf16-mixed",  # Uses BFloat16
    strategy="deepspeed_stage_2",
)

# FP16 Mixed Precision (for V100/older GPUs)
trainer = Trainer(
    accelerator="gpu",
    devices=8,
    precision="16-mixed",  # Uses Float16
    strategy="deepspeed_stage_2",
)

# Full Precision (baseline)
trainer = Trainer(
    accelerator="gpu",
    devices=8,
    precision="32",
)
```

**BF16 vs FP16:**
- **BF16 (BFloat16):** Better numerical stability, same dynamic range as FP32, no loss scaling needed
  - Recommended for: A100, H100, TPUs
  - Training speed: ~2-3x faster than FP32
  - Memory usage: ~50% of FP32

- **FP16 (Float16):** Requires loss scaling, potential underflow issues
  - Recommended for: V100, T4, older GPUs
  - Training speed: ~2-3x faster than FP32
  - Memory usage: ~50% of FP32
  - Requires careful loss scaling

---

### Gradient Accumulation

```python
from lightning.pytorch import Trainer

# Accumulate gradients over 4 batches
# Effective batch size = micro_batch_size × devices × accumulate_grad_batches
trainer = Trainer(
    accelerator="gpu",
    devices=8,
    strategy="deepspeed_stage_2",
    precision="bf16-mixed",
    accumulate_grad_batches=4,  # Gradient accumulation
)

# Example: micro_batch=8, devices=8, accumulation=4
# Effective batch size = 8 × 8 × 4 = 256
```

**When to use gradient accumulation:**
- Simulate larger batch sizes without OOM errors
- Limited GPU memory
- Maintain effective batch size when using smaller micro-batches
- Recommended for models that benefit from large batch training

**Trade-offs:**
- Slower training (4x accumulation = 4x slower per effective batch)
- More stable gradients
- Better memory efficiency

---

### Complete Training Example with All Features

```python
import lightning as L
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
import torch
from torch import nn

class GUIModel(LightningModule):
    def __init__(self, model_config):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.build_model(model_config)

    def build_model(self, config):
        # Your model architecture
        return nn.Sequential(...)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.functional.cross_entropy(outputs, labels)

        # Logging
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'])

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.functional.cross_entropy(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,
            weight_decay=0.01,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

# DeepSpeed Strategy
strategy = DeepSpeedStrategy(
    stage=2,
    offload_optimizer=False,
    allgather_bucket_size=5e8,
    reduce_bucket_size=5e8,
)

# Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints/',
    filename='gui-model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    monitor='val_loss',
    mode='min',
)

lr_monitor = LearningRateMonitor(logging_interval='step')

# Trainer
trainer = Trainer(
    # Hardware
    accelerator="gpu",
    devices=8,
    num_nodes=1,

    # Strategy
    strategy=strategy,
    precision="bf16-mixed",

    # Training
    max_epochs=100,
    accumulate_grad_batches=2,
    gradient_clip_val=1.0,

    # Logging and Checkpointing
    callbacks=[checkpoint_callback, lr_monitor],
    log_every_n_steps=50,

    # Performance
    deterministic=False,  # Set True for reproducibility
    benchmark=True,  # Enable cudnn.benchmark for speed
)

# Train
model = GUIModel(model_config={})
train_loader = DataLoader(...)  # Your data loader
val_loader = DataLoader(...)    # Your validation loader

trainer.fit(model, train_loader, val_loader)
```

---

### Activation Checkpointing (Gradient Checkpointing)

Activation checkpointing trades compute for memory by recomputing activations during backward pass.

```python
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DeepSpeedStrategy

# Enable activation checkpointing with DeepSpeed
strategy = DeepSpeedStrategy(
    stage=2,
    # Activation checkpointing config
    config={
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False,  # Set True for extreme memory savings
            "contiguous_memory_optimization": True,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
        }
    }
)

trainer = Trainer(
    accelerator="gpu",
    devices=8,
    strategy=strategy,
    precision="bf16-mixed",
)
```

**Manual gradient checkpointing in model:**

```python
import torch
from torch.utils.checkpoint import checkpoint

class GUITransformer(LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock() for _ in range(24)
        ])

    def forward(self, x):
        for layer in self.layers:
            # Use gradient checkpointing for each layer
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

**Memory vs Compute Trade-off:**
- Memory savings: 30-50% reduction in activation memory
- Compute cost: ~20-30% slower training
- Recommended for: Large models, limited GPU memory, pre-training

---

## 4. Optimization Strategies

### Efficient Data Loading

#### WebDataset

WebDataset is ideal for large-scale datasets stored in cloud storage.

```python
import webdataset as wds
from torch.utils.data import DataLoader

# Create WebDataset
dataset = wds.WebDataset(
    "gs://my-bucket/train-{000000..001000}.tar",
    shardshuffle=True,
)

# Add preprocessing pipeline
dataset = (
    dataset
    .shuffle(1000)
    .decode("pil")
    .to_tuple("jpg", "json")
    .map_tuple(preprocess_image, preprocess_label)
    .batched(32)
)

# Create DataLoader
loader = DataLoader(
    dataset,
    batch_size=None,  # Batching handled by WebDataset
    num_workers=4,
    pin_memory=True,
)
```

**Benefits:**
- 6x faster when loading from cloud storage with distributed training
- Eliminates network communication overhead through sharding
- Scales from local datasets to petascale
- Works seamlessly with PyTorch Lightning

**Best practices:**
- Shard data into tar archives (10-100 MB per shard)
- Use `.shuffle()` for randomization
- Enable `shardshuffle=True` for epoch-level shuffling
- Set `num_workers=4` or higher for parallel loading

---

#### FFCV (Fast Forward Computer Vision)

FFCV provides the fastest data loading through custom .beton format.

```python
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.transforms import ToTensor, ToTorchImage

# Create FFCV Loader
loader = Loader(
    '/path/to/dataset.beton',
    batch_size=256,
    num_workers=8,
    order=OrderOption.RANDOM,
    distributed=True,  # Enable for distributed training
    pipelines={
        'image': [
            SimpleRGBImageDecoder(),
            ToTensor(),
            ToTorchImage(),
        ],
        'label': [IntDecoder(), ToTensor()],
    }
)
```

**Performance:**
- 2-3x faster than WebDataset
- 5-10x faster than standard PyTorch DataLoader
- Page-based storage eliminates random read penalties
- Optimized for both local and network file systems

**Creating FFCV dataset:**

```python
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from torchvision.datasets import ImageFolder

# Write dataset to FFCV format
writer = DatasetWriter(
    '/path/to/output.beton',
    {
        'image': RGBImageField(max_resolution=256),
        'label': IntField(),
    }
)

dataset = ImageFolder('/path/to/imagenet')
writer.from_indexed_dataset(dataset)
```

**When to use:**
- Data loading is a bottleneck
- Training on local NVMe or fast network storage
- Maximum throughput is critical
- Computer vision workloads

---

#### Data Loading Best Practices

```python
from torch.utils.data import DataLoader

# Optimized DataLoader configuration
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Set to number of CPU cores / GPUs
    pin_memory=True,  # Faster host-to-device transfer
    persistent_workers=True,  # Keep workers alive between epochs
    prefetch_factor=2,  # Prefetch 2 batches per worker
)
```

**General Guidelines:**
- `num_workers`: Start with 4, increase if data loading is bottleneck
- `pin_memory=True`: Always enable for GPU training
- `persistent_workers=True`: Reduces worker spawn overhead
- `prefetch_factor`: 2-4 is optimal for most cases
- Use SSD/NVMe storage for local datasets
- For cloud storage: WebDataset with pre-sharding

---

### Gradient Checkpointing

Already covered in PyTorch Lightning section. Key points:

**Selective Checkpointing (Advanced):**

```python
class SelectiveCheckpointModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock() for _ in range(48)
        ])
        self.checkpoint_every_n = 4  # Checkpoint every 4 blocks

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i % self.checkpoint_every_n == 0:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x
```

**Trade-off matrix:**
- Checkpoint every layer: Max memory savings (~50%), slowest (+30% time)
- Checkpoint every 2 layers: Good balance (~35% savings, +20% time)
- Checkpoint every 4 layers: Minimal impact (~20% savings, +10% time)

---

### Flash Attention

Flash Attention 2 provides 2-4x speedup for attention computation with 50-80% memory reduction.

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

class FlashAttentionModel(LightningModule):
    def __init__(self):
        super().__init__()
        # Your transformer model

    def forward(self, x):
        # Enable Flash Attention backend
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = self.transformer(x)
        return output
```

**Using Flash Attention 2 directly:**

```python
from flash_attn import flash_attn_qkvpacked_func
import torch

def forward_with_flash_attn(qkv, causal=False):
    """
    qkv: (batch, seqlen, 3, num_heads, head_dim)
    """
    output = flash_attn_qkvpacked_func(
        qkv,
        dropout_p=0.0,
        causal=causal,
        return_attn_probs=False,
    )
    return output
```

**DistFlashAttn for Long Context (2024):**

For sequences > 100k tokens, use DistFlashAttn with rematerialization-aware checkpointing:

```python
# Conceptual - refer to DistFlashAttn paper for implementation
from dist_flash_attn import DistFlashAttention

attn = DistFlashAttention(
    dim=768,
    num_heads=12,
    use_rematerialization_checkpointing=True,
)
```

**Benefits:**
- 2-4x faster attention computation
- 50-80% memory reduction
- Enables longer sequence lengths
- No accuracy loss

**Requirements:**
- A100 or H100 GPUs (Ampere architecture or newer)
- CUDA 11.8+
- PyTorch 2.0+

---

### Throughput Optimization Tips

#### 1. Use Larger Batch Sizes

```python
# Find maximum batch size
def find_max_batch_size(model, device):
    batch_size = 2
    while True:
        try:
            dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
            model(dummy_input)
            batch_size *= 2
        except RuntimeError:  # OOM
            return batch_size // 2
```

#### 2. Optimize DataLoader Workers

```python
# Monitor data loading time
import time

for batch in loader:
    start = time.time()
    # Training step
    train_time = time.time() - start

    # If data loading takes > 10% of iteration time, increase num_workers
```

#### 3. Enable Compilation (PyTorch 2.0+)

```python
class CompiledModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.compile(
            YourModel(),
            mode="max-autotune",  # or "reduce-overhead"
        )
```

**Expected speedup:** 10-30% faster training

#### 4. Use Fused Optimizers

```python
# Standard AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Fused AdamW (faster)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    fused=True,  # Enable kernel fusion
)
```

#### 5. Benchmark Settings

```python
trainer = Trainer(
    benchmark=True,  # Enable cudnn.benchmark for speed
    deterministic=False,  # Disable for maximum speed
)
```

**Comprehensive optimization checklist:**
- [ ] Use BF16/FP16 mixed precision
- [ ] Enable Flash Attention 2
- [ ] Use gradient checkpointing for large models
- [ ] Optimize batch size (largest that fits in memory)
- [ ] Efficient data loading (FFCV or WebDataset)
- [ ] Set `num_workers=4` or higher
- [ ] Enable `pin_memory=True`
- [ ] Enable `persistent_workers=True`
- [ ] Use fused optimizers
- [ ] Enable `torch.compile()` (PyTorch 2.0+)
- [ ] Enable `trainer.benchmark=True`
- [ ] Use gradient accumulation if needed
- [ ] Profile with PyTorch Profiler to find bottlenecks

---

## 5. Cost and Infrastructure

### Cloud Provider Comparison (2024-2025)

#### H100 GPU Pricing (80GB)

| Provider | On-Demand ($/GPU-hr) | Spot/Preemptible ($/GPU-hr) | Notes |
|----------|---------------------|----------------------------|-------|
| **AWS EC2 P5** | $7.57 → $3.90 | $2.00 - $2.50 | Price dropped significantly |
| **GCP A3** | $11.06 → $3.00 | $1.50 - $2.00 | High variance by region |
| **Azure NC H100 v5** | $6.98 - $10.00 | $3.00 - $4.00 | Varies by region |
| **CUDO Compute** | $2.25 - $2.47 | N/A | Specialized AI cloud |
| **Lambda Labs** | $2.99 | N/A | Best value for H100 SXM |
| **CoreWeave** | $6.16 | $3.00 - $4.00 | Gaming infrastructure |
| **Paperspace** | $5.95 | N/A | Easy to use |

**Key Insights:**
- Specialized AI clouds are 40-60% cheaper than hyperscalers
- AWS and GCP have dramatically reduced prices in 2025
- Azure tends to be most expensive in premium regions

---

#### A100 GPU Pricing (80GB)

| Provider | On-Demand ($/GPU-hr) | Spot/Preemptible ($/GPU-hr) | Notes |
|----------|---------------------|----------------------------|-------|
| **AWS EC2 P4** | $2.00 - $3.00 | $0.80 - $1.20 | Widely available |
| **GCP A2** | $2.50 - $3.50 | $0.70 - $1.00 | 60-91% spot savings |
| **Azure NC A100 v4** | $2.80 - $4.00 | $1.00 - $1.50 | Limited availability |
| **Lambda Labs** | $1.10 | N/A | Best value |
| **CUDO Compute** | $1.20 - $1.50 | N/A | Good availability |
| **Vast.ai** | $0.50 - $1.00 | N/A | Marketplace pricing |

**Key Insights:**
- A100s now < $1/GPU-hr in open market
- 3-5x cheaper than H100s
- Excellent for training < 100B parameter models
- Spot instances offer 60-80% savings

---

### Spot Instance Strategies

#### Preemption Characteristics

| Provider | Preemption Warning | Max Runtime | Spot Discount |
|----------|-------------------|-------------|---------------|
| **AWS** | 2 minutes | No limit | 50-90% |
| **GCP** | 30 seconds | 24 hours (Preemptible) / No limit (Spot) | 60-91% |
| **Azure** | 30 seconds | No limit | 60-90% |

---

#### Strategy 1: Checkpointing

Essential for all spot instance training.

```python
from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints/',
    filename='model-{epoch:02d}-{step}',
    save_top_k=-1,  # Save all checkpoints
    every_n_train_steps=500,  # Checkpoint every 500 steps
    save_on_train_epoch_end=True,
)

trainer = Trainer(
    callbacks=[checkpoint_callback],
    # Resume from checkpoint
    ckpt_path='checkpoints/last.ckpt' if os.path.exists('checkpoints/last.ckpt') else None,
)
```

**Best practices:**
- Save checkpoints every 10-30 minutes
- Save to cloud storage (S3, GCS) for durability
- Use `save_on_train_epoch_end=True`
- Implement automatic resume logic

---

#### Strategy 2: Multi-Region Deployment

Spread training across multiple regions to reduce correlated preemptions.

```python
# Example: Launch jobs in multiple regions
regions = ['us-west1', 'us-central1', 'us-east1']

# Use Ray for cross-region orchestration
from ray import train

def training_function():
    # Your training code
    pass

# Ray handles cross-region coordination
trainer = TorchTrainer(
    training_function,
    scaling_config=ScalingConfig(
        num_workers=8,
        use_gpu=True,
        placement_strategy="SPREAD",  # Spread across failure domains
    )
)
```

**Benefits:**
- Reduces correlated preemptions (cross-region preemptions are decorrelated)
- Higher availability
- 2-3x lower preemption rate

---

#### Strategy 3: Hybrid On-Demand + Spot

```yaml
# Example: GKE node pool configuration
nodePools:
  - name: on-demand-pool
    initialNodeCount: 2
    config:
      machineType: a2-highgpu-8g

  - name: spot-pool
    initialNodeCount: 8
    config:
      machineType: a2-highgpu-8g
      spot: true
```

**Strategy:**
- 20-30% on-demand instances (critical workers)
- 70-80% spot instances (distributed workers)
- Master/coordinator always on-demand

---

### Cost Estimates for Different Scales

#### Small Scale: Fine-tuning 7B Model

**Configuration:**
- Model: 7B parameters (LLaMA-2 7B)
- Hardware: 8x A100 80GB
- Duration: 24 hours
- Dataset: 100GB

**Cost Breakdown:**

| Item | AWS On-Demand | AWS Spot | Lambda Labs | Notes |
|------|---------------|----------|-------------|-------|
| Compute (8 GPU × 24hr) | $576 | $230 | $211 | Largest cost component |
| Storage (100GB × 24hr) | $2 | $2 | $2 | S3/equivalent |
| Network egress | $10 | $10 | $5 | Downloading checkpoints |
| **Total** | **$588** | **$242** | **$218** | 59% savings with spot |

**Recommendations:**
- Use Lambda Labs or spot instances
- Enable checkpointing every 30 minutes
- Use S3 for checkpoint storage

---

#### Medium Scale: Pre-training 30B Model

**Configuration:**
- Model: 30B parameters
- Hardware: 64x A100 80GB
- Duration: 7 days (168 hours)
- Dataset: 2TB

**Cost Breakdown:**

| Item | AWS On-Demand | GCP Spot | CUDO Compute | Notes |
|------|---------------|----------|--------------|-------|
| Compute (64 GPU × 168hr) | $32,256 | $7,526 | $12,902 | 7 days of training |
| Storage (2TB × 168hr) | $340 | $284 | $300 | Approximate |
| Network egress | $200 | $150 | $100 | Checkpoint syncing |
| **Total** | **$32,796** | **$7,960** | **$13,302** | 76% savings with GCP spot |

**Recommendations:**
- Use GCP spot instances with multi-region deployment
- Implement aggressive checkpointing (every 15 minutes)
- Use ZeRO-2 for optimal training speed
- Consider CUDO Compute for price/performance balance

---

#### Large Scale: Pre-training 70B+ Model

**Configuration:**
- Model: 70B parameters
- Hardware: 256x H100 80GB
- Duration: 30 days (720 hours)
- Dataset: 10TB

**Cost Breakdown:**

| Item | AWS On-Demand | GCP Spot | Lambda Labs | Notes |
|------|---------------|----------|-------------|-------|
| Compute (256 GPU × 720hr) | $719,616 | $276,480 | $551,424 | Month of training |
| Storage (10TB × 720hr) | $1,440 | $1,200 | $1,200 | S3/GCS |
| Network egress | $2,000 | $1,500 | $1,000 | Multi-TB checkpoints |
| **Total** | **$723,056** | **$279,180** | **$553,624** | 61% savings with GCP spot |

**Recommendations:**
- Mandatory: Use spot instances with multi-region strategy
- Use ZeRO-2 or ZeRO-3 depending on memory constraints
- Implement distributed checkpointing
- Use FFCV or WebDataset for data loading
- Consider 3-month reserved instances if training > 1 month

---

### Infrastructure Best Practices

#### Network Optimization

```python
# Enable NCCL optimizations for distributed training
import os

# Set NCCL environment variables
os.environ['NCCL_IB_DISABLE'] = '0'  # Enable InfiniBand if available
os.environ['NCCL_IB_GID_INDEX'] = '3'
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # Set network interface
os.environ['NCCL_DEBUG'] = 'INFO'  # Enable debug logging

# For AWS EFA (Elastic Fabric Adapter)
os.environ['FI_PROVIDER'] = 'efa'
os.environ['FI_EFA_USE_DEVICE_RDMA'] = '1'
```

---

#### Monitoring and Cost Tracking

```python
# Track GPU utilization and costs
from lightning.pytorch.callbacks import Callback
import time

class CostTracker(Callback):
    def __init__(self, cost_per_gpu_hour=2.5, num_gpus=8):
        self.cost_per_gpu_hour = cost_per_gpu_hour
        self.num_gpus = num_gpus
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed_hours = (time.time() - self.start_time) / 3600
        total_cost = elapsed_hours * self.cost_per_gpu_hour * self.num_gpus

        trainer.logger.log_metrics({
            'total_cost_usd': total_cost,
            'cost_per_sample': total_cost / trainer.global_step,
        })

# Use in trainer
trainer = Trainer(
    callbacks=[CostTracker(cost_per_gpu_hour=2.5, num_gpus=8)]
)
```

---

## 6. Monitoring and Debugging Distributed Training

### TensorBoard Integration

```python
from lightning.pytorch.loggers import TensorBoardLogger

# Create TensorBoard logger
logger = TensorBoardLogger(
    save_dir='logs/',
    name='gui_training',
    version='v1',
)

trainer = Trainer(
    logger=logger,
    log_every_n_steps=50,
)

# In training step
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)

    # Log scalars
    self.log('train_loss', loss)
    self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'])

    # Log images every 100 steps
    if batch_idx % 100 == 0:
        self.logger.experiment.add_images(
            'predictions',
            predictions,
            global_step=self.global_step
        )

    return loss
```

**Launch TensorBoard:**
```bash
tensorboard --logdir logs/ --port 6006
```

---

### Weights & Biases (WandB) Integration

```python
from lightning.pytorch.loggers import WandbLogger
import wandb

# Initialize WandB logger
wandb_logger = WandbLogger(
    project='gui-training',
    name='experiment-1',
    config={
        'learning_rate': 1e-4,
        'batch_size': 256,
        'num_gpus': 8,
    },
    sync_tensorboard=True,  # Sync TensorBoard logs
)

trainer = Trainer(
    logger=wandb_logger,
    log_every_n_steps=50,
)

# Advanced logging in training step
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)

    # Log to WandB
    self.log('train_loss', loss)

    # Log custom charts
    if batch_idx % 100 == 0:
        wandb.log({
            'predictions': wandb.Image(predictions),
            'confusion_matrix': wandb.plot.confusion_matrix(
                probs=probs,
                y_true=labels,
                class_names=class_names,
            )
        })

    return loss
```

---

### Distributed Training Logging Best Practices

```python
class DistributedModel(LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        # sync_dist=True aggregates metrics across all GPUs
        self.log('train_loss', loss, sync_dist=True, prog_bar=True)

        # rank_zero_only: log only from rank 0 (avoid duplicate logs)
        self.log('rank_zero_metric', value, rank_zero_only=True)

        return loss
```

**WandB distributed training strategies:**

```python
import wandb
import os

# Strategy 1: Single run for all processes (recommended)
wandb_logger = WandbLogger(
    project='distributed-training',
    group='experiment-1',  # Group related runs
    job_type='train',
)

# Strategy 2: Track each process separately
if int(os.environ.get('LOCAL_RANK', 0)) == 0:
    # Only rank 0 logs
    wandb_logger = WandbLogger(project='distributed-training')
else:
    wandb_logger = None
```

---

### DeepSpeed Monitor Integration

```json
{
  "tensorboard": {
    "enabled": true,
    "output_path": "logs/tensorboard/",
    "job_name": "train_deepspeed"
  },
  "wandb": {
    "enabled": true,
    "team": "my-team",
    "project": "my-project",
    "group": "experiment-1"
  }
}
```

---

### Debugging Distributed Training

#### Common Issues and Solutions

**1. Gradient synchronization issues:**

```python
# Check if gradients are synchronized
def on_before_optimizer_step(self, optimizer):
    # Log gradient norms for debugging
    total_norm = 0.0
    for p in self.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    self.log('grad_norm', total_norm, rank_zero_only=True)
```

**2. NCCL debugging:**

```bash
# Enable verbose NCCL logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Check NCCL version
python -c "import torch; print(torch.cuda.nccl.version())"
```

**3. Deadlock detection:**

```python
# Set timeout for distributed operations
import torch.distributed as dist

# Initialize with timeout
dist.init_process_group(
    backend='nccl',
    timeout=timedelta(minutes=30),
)
```

**4. Memory profiling:**

```python
from lightning.pytorch.callbacks import Callback
import torch

class MemoryProfiler(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 100 == 0:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9

            print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

            trainer.logger.log_metrics({
                'gpu_memory_allocated_gb': allocated,
                'gpu_memory_reserved_gb': reserved,
            })
```

---

### Performance Profiling

```python
from torch.profiler import profile, record_function, ProfilerActivity

def profile_training():
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("training_iteration"):
            # Your training step
            loss = model(batch)
            loss.backward()
            optimizer.step()

    # Print profiling results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Export to Chrome trace
    prof.export_chrome_trace("trace.json")
```

**Analyze with TensorBoard:**
```bash
tensorboard --logdir=./logs --bind_all
```

---

### Best Practices Summary

**Logging:**
- Use WandB or TensorBoard for experiment tracking
- Always set `sync_dist=True` for distributed metrics
- Log system metrics (GPU utilization, memory, throughput)
- Use `rank_zero_only=True` to avoid duplicate logs
- Group related experiments in WandB

**Debugging:**
- Enable NCCL debug logging for communication issues
- Profile with PyTorch Profiler to find bottlenecks
- Monitor gradient norms for training stability
- Track GPU memory usage
- Set distributed timeouts to detect deadlocks

**Cost Tracking:**
- Log cost metrics (total cost, cost per sample)
- Monitor GPU utilization (should be > 80%)
- Track throughput (samples/second)
- Use spot instances with checkpointing
- Implement automatic cost alerts

---

## Conclusion

This document provides a comprehensive overview of distributed training frameworks and strategies for training large-scale models. Key takeaways:

1. **Framework Selection:**
   - Use PyTorch Lightning + DeepSpeed for large models (500M+ parameters)
   - Use Hugging Face Accelerate for flexibility and ease of use
   - Use PyTorch FSDP for native PyTorch workflows
   - Use Ray Train for multi-framework pipelines

2. **DeepSpeed ZeRO:**
   - ZeRO-2 for pre-training (10-20B models on 128+ GPUs)
   - ZeRO-3 for fine-tuning (10-20B models on limited GPUs)
   - ZeRO-3 Offload for single-GPU training of large models

3. **Optimizations:**
   - Enable BF16 mixed precision on A100/H100
   - Use Flash Attention 2 for 2-4x speedup
   - Implement gradient checkpointing for memory savings
   - Use FFCV or WebDataset for efficient data loading

4. **Cost:**
   - Specialized AI clouds are 40-60% cheaper than hyperscalers
   - Spot instances offer 60-90% savings
   - Use multi-region deployment for reliability
   - Checkpoint frequently (every 10-30 minutes)

5. **Monitoring:**
   - Use WandB for comprehensive experiment tracking
   - Enable DeepSpeed Monitor for unified logging
   - Profile with PyTorch Profiler to optimize performance
   - Track costs and GPU utilization

**Next Steps:**
- Review AGENT_HANDOFF.md for project-specific requirements
- Set up training infrastructure with chosen framework
- Implement efficient data loading pipeline
- Configure monitoring and cost tracking
- Start with small-scale experiments before scaling up

---

**References:**
- [PyTorch Lightning DeepSpeed Documentation](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html)
- [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/)
- [Hugging Face Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [Ray Train Documentation](https://docs.ray.io/en/latest/train/train.html)
- [Flash Attention GitHub](https://github.com/Dao-AILab/flash-attention)
- [FFCV Documentation](https://docs.ffcv.io/)
- [WebDataset GitHub](https://github.com/webdataset/webdataset)
