# Research Findings - qontinui-train

**Project**: Foundation Model Training for GUI Understanding
**Repository**: qontinui-train
**Owner**: Joshua Spinak
**Last Updated**: 2025-11-14

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Selection](#architecture-selection)
3. [Pre-training Methods](#pre-training-methods)
4. [Distributed Training Strategy](#distributed-training-strategy)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Cost Estimates](#cost-estimates)
7. [References](#references)

---

## Executive Summary

This document consolidates research findings for building a foundation model for GUI understanding trained from scratch on millions of examples. Based on comprehensive analysis of current state-of-the-art architectures, pre-training methods, and distributed training frameworks, we recommend the following approach:

### Recommended Architecture Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Backbone** | Swin Transformer Base | Linear O(N) complexity, hierarchical features, proven for high-res images |
| **Detection Head** | Deformable-DETR | Set-based prediction, 10× faster convergence, no NMS required |
| **Pre-training** | Masked Autoencoder (MAE) | 3× faster than contrastive learning, preserves layout structure |
| **Training Framework** | PyTorch Lightning + DeepSpeed ZeRO-2 | 8× memory reduction, clean abstractions, production-ready |
| **Data Pipeline** | FFCV or WebDataset | 2-3× faster data loading, essential for large-scale training |
| **Experiment Tracking** | Weights & Biases | Industry standard, distributed training support |

### Key Performance Targets

- **Detection mAP**: > 0.95 on standard benchmarks
- **Zero-shot transfer**: > 0.70 mAP on unseen applications
- **Few-shot (10 examples)**: > 0.85 mAP
- **Inference speed**: < 30ms per frame
- **Training time**: 2-4 days for 1M examples on 8× A100

### Scale Strategy

1. **Phase 1 (Proof of Concept)**: 100K examples, 2-3 days, ~$200
2. **Phase 2 (Small Scale)**: 1M examples, 5-7 days, ~$800
3. **Phase 3 (Large Scale)**: 10M+ examples, 3-4 weeks, ~$8,000-$12,000

---

## Architecture Selection

### Vision Transformer Variants Evaluated

After analyzing ViT, Swin, DeiT, and DETR-based architectures, we recommend **Swin Transformer** for the following reasons:

#### 1. Swin Transformer (RECOMMENDED)

**Advantages**:
- **Linear complexity O(N)** vs quadratic O(N²) for standard ViT
- Handles high-resolution screenshots (up to 1024×1024) efficiently
- Hierarchical architecture (4×4 → 32×32 patches) naturally captures GUI structure
- Outperforms ViT by +2.4% accuracy with better throughput
- Proven in production: Google's ScreenAI, Microsoft's OmniParser

**Architecture**:
```
Input: 1024×1024 screenshot
├─ Patch Embedding: 4×4 patches → 96-dim (256×256 tokens)
├─ Stage 1: Swin blocks, window size 7×7
├─ Stage 2: Downsample + Swin blocks (128×128 tokens)
├─ Stage 3: Downsample + Swin blocks (64×64 tokens)
└─ Stage 4: Downsample + Swin blocks (32×32 tokens)

Output: Multi-scale features [128×128, 64×64, 32×32] with [192, 384, 768] channels
```

**Model Sizes**:
- **Swin-Tiny**: 28M params, ~5 GFLOPs, suitable for initial experiments
- **Swin-Small**: 50M params, ~9 GFLOPs, balanced performance
- **Swin-Base**: 88M params, ~15 GFLOPs, recommended for production
- **Swin-Large**: 197M params, ~35 GFLOPs, for maximum accuracy

**Memory Requirements** (Swin-Base, mixed precision):
- Training: ~12GB GPU at 1024×1024, batch size 2
- Inference: ~2GB GPU
- Gradient checkpointing: 30-50% memory savings

#### 2. Standard ViT (Alternative)

**Advantages**:
- Simpler architecture, easier to understand and modify
- Extensive pre-training methods (MAE, CLIP, DeiT)
- Well-supported by timm and transformers libraries
- Global receptive field from first layer

**Disadvantages**:
- Quadratic complexity O(N²) limits max resolution to 512×512
- Higher computational cost for equivalent performance
- No hierarchical features (may be less suitable for GUI structure)

**When to use**: If simplicity is prioritized over efficiency, or for comparison baselines

#### 3. Detection Head Design

We recommend **Deformable-DETR** for the detection head:

**Architecture**:
```
Swin Backbone Features → Deformable Transformer Encoder → Object Queries → Predictions
                          (Multi-scale deformable attention)

Object Queries: 100-300 learnable embeddings (one per potential GUI element)
Decoder: 6 Deformable Transformer layers
Output: {class, bbox} for each query (set-based prediction)
```

**Advantages over alternatives**:
- **10× faster convergence** than standard DETR
- No non-maximum suppression (NMS) needed
- Multi-scale deformable attention focuses on key points
- State-of-the-art variants (NAN-DETR) achieve 50.1% AP on COCO

**Loss Function**:
```python
Total Loss = λ_cls · Focal Loss(classification)
           + λ_bbox · L1 Loss(bboxes)
           + λ_giou · GIoU Loss(bboxes)

Recommended: λ_cls=2.0, λ_bbox=5.0, λ_giou=2.0
```

**Implementation**: Use DETR codebase from Facebook Research, adapt for Swin backbone

### Final Architecture Recommendation

**Model**: Swin-Base + Deformable-DETR
- **Input**: 800×1024 screenshots (or tile-based for larger)
- **Backbone**: Swin-B pretrained on ImageNet-22K (optional) or trained from scratch
- **Detection**: Deformable-DETR with 200 object queries
- **Total Parameters**: ~130M (88M backbone + 42M detection head)
- **Training Memory**: ~16GB per GPU with batch size 2 (mixed precision)

**Configuration**:
```yaml
model:
  backbone: swin_base_patch4_window7_224
  pretrained: false  # Train from scratch
  input_size: [800, 1024]
  num_classes: 50  # GUI element types
  num_queries: 200

training:
  batch_size: 2  # Per GPU
  accumulate_grad_batches: 8  # Effective batch size: 128 on 8 GPUs
  precision: "16-mixed"
  max_epochs: 300
```

---

## Pre-training Methods

### Methods Evaluated

We evaluated three self-supervised pre-training approaches:

1. **Masked Autoencoder (MAE)** - RECOMMENDED
2. **Contrastive Learning (MoCo, SimCLR)**
3. **Multi-modal (CLIP-style)**

### 1. Masked Autoencoder (MAE) - RECOMMENDED

**How it works**:
1. Randomly mask 75% of image patches
2. Encode visible patches with ViT/Swin encoder
3. Decode to reconstruct masked patches
4. Minimize reconstruction loss (MSE)

**Why MAE for GUI Understanding**:
- **3× faster training** than contrastive learning (only encodes 25% of patches)
- **Preserves layout structure** - critical for GUI understanding
- **Simple implementation** - single forward pass, no negative sampling
- **Works well with limited data** - effective with 100K-1M unlabeled screenshots

**Expected Performance Gains**:
- **+10-15% mAP** on GUI detection tasks vs. training from scratch
- **+5-8% mAP** vs. ImageNet pre-training
- Especially beneficial for:
  - Small GUI elements (buttons, icons): +15-20%
  - Complex layouts: +12-18%
  - Zero-shot transfer: +10-15%

**Training Configuration**:
```yaml
mae_pretrain:
  model:
    encoder: swin_base
    decoder: lightweight_transformer  # 8 layers, 512-dim
    mask_ratio: 0.75
    reconstruction_target: pixels  # or "features"

  training:
    epochs: 400
    batch_size: 256  # Per GPU with gradient accumulation
    learning_rate: 1.5e-4
    warmup_epochs: 40
    optimizer: AdamW
    weight_decay: 0.05

  data:
    unlabeled_screenshots: 1_000_000
    augmentations: [random_crop, color_jitter]
    resolution: 224  # Lower res for pre-training

  compute:
    gpus: 8  # A100 80GB
    estimated_time: "2-3 days"
    estimated_cost: "$300-400"
```

**Implementation**:
- Use Facebook Research MAE codebase as reference
- Adapt for Swin Transformer (replace ViT patch embedding)
- Train on unlabeled GUI screenshots (no annotations needed)
- Save encoder weights for fine-tuning

### 2. Contrastive Learning (MoCo, SimCLR)

**When to use**:
- Large compute available (32+ GPUs)
- Need similarity embeddings for retrieval tasks
- Have massive unlabeled datasets (10M+)

**Challenges for GUIs**:
- Requires careful augmentation design (standard augmentations may destroy layout)
- Massive batch sizes needed (4K-8K for SimCLR)
- 3× slower training than MAE
- More complex implementation

**Augmentations for GUI screenshots**:
```python
# GOOD: Preserve layout structure
augmentations = [
    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
]

# AVOID: May destroy semantic meaning
avoid = [
    RandomRotation,  # Breaks reading direction
    RandomPerspective,  # Distorts UI structure
    RandomAffine,  # May misalign elements
]
```

**Cost Comparison**:
- MAE: $300-400 for 1M examples, 2-3 days
- MoCo: $800-1,200 for 1M examples, 4-6 days
- SimCLR: $2,000-3,000 for 1M examples, 7-10 days

### 3. Multi-modal Pre-training (CLIP-style)

**Approach**: Align vision and text representations
- Vision: Screenshots
- Text: DOM text, accessibility labels, UI element descriptions

**Advantages**:
- **Best semantic understanding** (+12-18% mAP when DOM data available)
- **Zero-shot capabilities** (describe element in natural language)
- **Interpretable embeddings** (similarity search, retrieval)

**Challenges**:
- Requires paired (image, text) data - not always available
- More complex data collection pipeline
- Longer training time than MAE

**When to use**:
- Building a multi-modal GUI understanding system
- Have DOM/accessibility data for training screenshots
- Zero-shot element search is a key requirement

**Expected Performance**:
- Zero-shot element detection by description: 65-70% recall
- Semantic similarity for UI retrieval: 80-85% top-5 accuracy
- Standard detection: +12-18% mAP over MAE when text data is high-quality

### Pre-training Strategy Recommendation

**Phase 1: MAE Pre-training** (Weeks 1-2)
1. Collect 1M unlabeled GUI screenshots (web, mobile, desktop)
2. Train MAE with Swin-Base encoder for 400 epochs
3. Cost: ~$400, Time: 2-3 days on 8× A100

**Phase 2: Fine-tuning** (Weeks 3-4)
1. Prepare 10K-50K labeled detection samples
2. Fine-tune encoder + detection head for 100 epochs
3. Cost: ~$200, Time: 1-2 days

**Phase 3: Evaluation** (Week 5)
1. Benchmark on test sets
2. Compare to ImageNet pre-training baseline
3. Measure zero-shot transfer capabilities

**Total Cost**: ~$600 for full pipeline (vs $2,000+ for contrastive learning)

---

## Distributed Training Strategy

### Framework Selection: PyTorch Lightning + DeepSpeed ZeRO-2

After evaluating PyTorch Lightning, Hugging Face Accelerate, PyTorch FSDP, and Ray Train, we recommend **PyTorch Lightning + DeepSpeed ZeRO-2** for the following reasons:

#### Why PyTorch Lightning?

**Advantages**:
- Clean training abstractions (no boilerplate code)
- Built-in support for distributed training, mixed precision, gradient accumulation
- Excellent debugging and profiling tools
- Easy integration with DeepSpeed, FSDP
- Strong community and documentation

**Disadvantages**:
- Adds a layer of abstraction (may obscure low-level details)
- Learning curve for Lightning-specific patterns

#### Why DeepSpeed ZeRO-2?

**ZeRO Stage Comparison**:

| Stage | Memory Reduction | Use Case | Best For |
|-------|-----------------|----------|----------|
| **ZeRO-1** | 4× | Optimizer state partitioning | Small models (<1B params) |
| **ZeRO-2** | 8× | + Gradient partitioning | **Pre-training (recommended)** |
| **ZeRO-3** | Linear with GPUs | + Parameter partitioning | Fine-tuning, very large models |

**Why ZeRO-2 for our use case**:
- **8× memory reduction** allows larger batch sizes
- **No communication overhead** during forward pass (unlike ZeRO-3)
- **Faster training** than ZeRO-3 for models under 1B parameters
- **Optimal for Swin-Base (88M params)** on 8× A100 GPUs

**Configuration**:
```yaml
trainer:
  accelerator: gpu
  devices: 8
  strategy:
    type: deepspeed
    stage: 2  # ZeRO-2
    offload_optimizer: false  # Keep on GPU for speed
    allgather_bucket_size: 2e8
    reduce_bucket_size: 2e8
  precision: "16-mixed"  # BF16 on A100/H100, FP16 on V100
  accumulate_grad_batches: 8  # Effective batch size scaling
```

### Optimization Strategies

#### 1. Mixed Precision Training

**BF16 vs FP16**:
- **BF16**: Better numerical stability, no loss scaling needed, available on A100/H100
- **FP16**: 2× speedup, 50% memory savings, requires loss scaling, prone to overflow

**Recommendation**: Use **BF16** on A100/H100, **FP16** on older GPUs

**Expected Gains**:
- 2-3× faster training
- 50% memory savings
- Minimal accuracy impact (<0.5% mAP)

#### 2. Efficient Data Loading

**FFCV vs WebDataset**:

| Feature | FFCV | WebDataset |
|---------|------|------------|
| **Speed** | 2-3× faster than PyTorch | 1.5-2× faster than PyTorch |
| **Setup** | Requires dataset conversion | Works with tar archives |
| **Random Access** | Excellent (memory-mapped) | Sequential (streaming) |
| **Best For** | Fixed datasets | Very large datasets (>10TB) |

**Recommendation**:
- **Phase 1-2 (100K-1M examples)**: Use FFCV for maximum speed
- **Phase 3 (10M+ examples)**: Use WebDataset for streaming from cloud storage

**FFCV Configuration**:
```python
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, JSONField

# Convert dataset to FFCV format
writer = DatasetWriter(
    'data/train.beton',
    {
        'image': RGBImageField(),
        'annotations': JSONField(),
    },
    num_workers=32
)
writer.from_indexed_dataset(dataset)

# Training: 2-3× faster data loading
from ffcv.loader import Loader
loader = Loader('data/train.beton', batch_size=32, num_workers=8)
```

#### 3. Gradient Checkpointing

**Trade-off**:
- **Memory savings**: 30-50%
- **Speed cost**: 20-30% slower

**When to use**:
- GPU memory constrained
- Want larger batch sizes
- Training large models (>1B params)

**Implementation**:
```python
# Enable in Swin Transformer
model = SwinTransformer(use_checkpoint=True)  # Checkpoint every Swin block
```

#### 4. Flash Attention 2

**Benefits**:
- 2-4× speedup for attention layers
- 50-80% memory reduction
- Exact attention (not approximate)

**Requirements**:
- A100 or H100 GPUs (requires Tensor Cores)
- CUDA 11.4+
- PyTorch 2.0+

**Implementation**:
```python
# Install: pip install flash-attn --no-build-isolation
from flash_attn import flash_attn_qkvpacked_func

# Replace standard attention in Swin blocks
# Expected speedup: 2-3× for Swin Transformer
```

### Cost Optimization

#### Cloud Provider Comparison (2024-2025 Prices)

**H100 GPUs** (80GB, best performance):
| Provider | On-Demand | Spot | Best For |
|----------|-----------|------|----------|
| AWS | $3.00/GPU-hr | $1.50-2.00/GPU-hr | Enterprise, reliability |
| GCP | $3.50/GPU-hr | $1.80-2.50/GPU-hr | Integration with GCP services |
| Lambda Labs | $2.25/GPU-hr | N/A | Best value for AI workloads |
| CUDO Compute | $2.50/GPU-hr | $1.50/GPU-hr | Flexible, spot-friendly |

**A100 GPUs** (80GB, great value):
| Provider | On-Demand | Spot | Best For |
|----------|-----------|------|----------|
| AWS | $2.00/GPU-hr | $0.70-1.20/GPU-hr | Mature spot market |
| Lambda Labs | $1.10/GPU-hr | N/A | **Best value overall** |
| RunPod | $1.20/GPU-hr | $0.80/GPU-hr | Easy setup, good docs |

#### Spot Instance Strategy

**Preemption Handling**:
1. **Checkpointing**: Save every 10-30 minutes
2. **Multi-region**: Spread across regions (2-3× lower preemption)
3. **Hybrid**: 20-30% on-demand (critical workers) + 70-80% spot

**Implementation**:
```python
# PyTorch Lightning auto-checkpointing
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints/',
    filename='{epoch}-{step}',
    every_n_train_steps=500,  # Checkpoint every 500 steps (~10 min)
    save_top_k=-1,  # Keep all checkpoints
)
```

**Expected Savings**:
- Spot vs on-demand: 60-75% cost reduction
- Multi-region spot: Additional 10-20% uptime improvement
- Total cost reduction: ~65-80% with proper spot strategy

### Training Timeline and Cost Estimates

#### Phase 1: Proof of Concept (100K examples)

**Configuration**:
- Model: Swin-Small + Deformable-DETR
- GPUs: 4× A100 (40GB)
- Batch size: 16 (4 per GPU)
- Epochs: 100
- Estimated time: 24-36 hours

**Cost**:
- Lambda Labs (on-demand): 4 × $1.10/hr × 30 hrs = **$132**
- AWS Spot: 4 × $0.80/hr × 30 hrs = **$96**

#### Phase 2: MAE Pre-training (1M examples)

**Configuration**:
- Model: Swin-Base MAE
- GPUs: 8× A100 (80GB)
- Batch size: 256 (32 per GPU)
- Epochs: 400
- Estimated time: 48-72 hours

**Cost**:
- Lambda Labs: 8 × $1.10/hr × 60 hrs = **$528**
- AWS Spot: 8 × $0.80/hr × 60 hrs = **$384**

#### Phase 3: Detection Fine-tuning (1M examples)

**Configuration**:
- Model: Swin-Base + Detection Head
- GPUs: 8× A100 (80GB)
- Batch size: 128 (16 per GPU)
- Epochs: 100
- Estimated time: 24-36 hours

**Cost**:
- Lambda Labs: 8 × $1.10/hr × 30 hrs = **$264**
- AWS Spot: 8 × $0.80/hr × 30 hrs = **$192**

#### Phase 4: Large-Scale Training (10M examples)

**Configuration**:
- Model: Swin-Large + Detection Head
- GPUs: 16× A100 (80GB) or 8× H100 (80GB)
- Batch size: 256
- Epochs: 300
- Estimated time: 7-10 days

**Cost**:
- 16× A100 (Lambda): 16 × $1.10/hr × 200 hrs = **$3,520**
- 8× H100 (Lambda): 8 × $2.25/hr × 140 hrs = **$2,520**
- AWS Spot (16× A100): 16 × $0.80/hr × 200 hrs = **$2,560**

**Recommendation**: Use 8× H100 for best performance/cost ratio at scale

### Monitoring and Debugging

**Weights & Biases Integration**:
```python
from pytorch_lightning.loggers import WandbLogger

logger = WandbLogger(
    project="qontinui-train",
    name="swin_base_mae_pretrain",
    log_model=True
)

trainer = Trainer(logger=logger, ...)
```

**Key Metrics to Monitor**:
- Training: loss, learning rate, gradient norm
- Validation: mAP, mAP@50, mAP@75, per-class AP
- System: GPU utilization, memory usage, throughput (samples/sec)
- Distributed: communication overhead, load balancing

**Profiling**:
```python
# PyTorch Lightning profiler
trainer = Trainer(
    profiler="advanced",  # or "simple", "pytorch"
    detect_anomaly=True,  # Detect NaN/Inf
)
```

---

## Implementation Roadmap

### Week 1-2: Infrastructure Setup

**Tasks**:
1. ✅ Clone reference repositories (timm, MAE, DeepSpeed, etc.)
2. ✅ Create code structure (models, training, data, evaluation)
3. ✅ Set up experiment tracking (Weights & Biases)
4. Set up cloud GPU instances (Lambda Labs or AWS)
5. Test distributed training with small model and small dataset

**Deliverables**:
- Working multi-GPU training pipeline
- Experiment tracking configured
- Baseline model trained on 10K examples

**Estimated Cost**: $50-100 (testing and debugging)

### Week 3-4: Data Generation

**Tasks**:
1. Implement synthetic UI generation pipeline
   - Web UIs (headless browser rendering)
   - Mobile UIs (Android/iOS simulators)
   - Desktop UIs (Qt/GTK rendering)
2. Generate 100K diverse synthetic screenshots
3. Validate data quality and diversity
4. Convert to FFCV format for fast loading

**Deliverables**:
- 100K synthetic screenshots with annotations
- Data generation scripts
- Data quality report

**Estimated Cost**: $100-200 (compute for data generation)

### Week 5-6: MAE Pre-training

**Tasks**:
1. Implement MAE pre-training pipeline
2. Collect 1M unlabeled GUI screenshots
3. Train Swin-Base MAE for 400 epochs
4. Validate reconstruction quality
5. Save encoder weights

**Deliverables**:
- Pre-trained Swin-Base encoder
- Pre-training logs and metrics
- Reconstruction visualizations

**Estimated Cost**: $400-600

### Week 7-8: Detection Fine-tuning

**Tasks**:
1. Implement Deformable-DETR detection head
2. Prepare 50K labeled detection samples
3. Fine-tune MAE encoder + detection head
4. Hyperparameter tuning
5. Evaluate on validation set

**Deliverables**:
- Trained detection model
- Hyperparameter sweep results
- Validation metrics (mAP, per-class AP)

**Estimated Cost**: $300-500

### Week 9-10: Evaluation and Transfer

**Tasks**:
1. Create evaluation benchmarks
   - Standard detection (Rico, CLAY datasets)
   - Zero-shot transfer (unseen applications)
   - Few-shot learning (10, 50, 100 examples)
2. Compare to baselines (ImageNet pre-trained models)
3. Inference speed benchmarking
4. Ablation studies

**Deliverables**:
- Comprehensive evaluation report
- Comparison to baselines
- Ablation study results
- Speed benchmarks

**Estimated Cost**: $100-200

### Week 11-12: Scaling and Optimization

**Tasks**:
1. Scale to 10M examples
2. Train larger model (Swin-Large)
3. Optimize inference (quantization, TensorRT)
4. Export model for production use

**Deliverables**:
- Foundation model trained on 10M examples
- Optimized inference engine
- Model checkpoints and documentation

**Estimated Cost**: $2,500-3,500

**Total Timeline**: 12 weeks
**Total Estimated Cost**: $3,450-$5,100

---

## Cost Estimates Summary

### By Phase

| Phase | Examples | GPU Type | Duration | Cost (Lambda) | Cost (AWS Spot) |
|-------|----------|----------|----------|---------------|-----------------|
| **Phase 1**: POC | 100K | 4× A100 | 30 hrs | $132 | $96 |
| **Phase 2**: MAE Pre-train | 1M | 8× A100 | 60 hrs | $528 | $384 |
| **Phase 3**: Fine-tune | 1M | 8× A100 | 30 hrs | $264 | $192 |
| **Phase 4**: Large-scale | 10M | 8× H100 | 140 hrs | $2,520 | $1,792 |
| **Testing & Debug** | - | Various | 40 hrs | $300 | $200 |
| **Total** | 10M+ | - | 300 hrs | **$3,744** | **$2,664** |

### Cost Optimization Strategies

1. **Use spot instances**: 60-75% cost reduction
2. **Start small, scale gradually**: Validate approach before large-scale training
3. **Efficient data loading (FFCV)**: 2-3× faster training = 50-66% time savings
4. **Mixed precision (BF16)**: 2× faster training = 50% cost savings
5. **Gradient accumulation**: Train on fewer GPUs with larger effective batch size

**Optimized Total Cost**: $1,500-$2,500 with all strategies applied

---

## References

### Papers

**Architecture**:
1. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT, 2020)
2. Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (2021)
3. Carion et al., "End-to-End Object Detection with Transformers" (DETR, 2020)
4. Zhu et al., "Deformable DETR: Deformable Transformers for End-to-End Object Detection" (2021)

**Pre-training**:
5. He et al., "Masked Autoencoders Are Scalable Vision Learners" (MAE, 2022)
6. Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR, 2020)
7. He et al., "Momentum Contrast for Unsupervised Visual Representation Learning" (MoCo, 2020)
8. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, 2021)

**Distributed Training**:
9. Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (DeepSpeed, 2020)
10. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)

**GUI Understanding**:
11. Baechler et al., "ScreenAI: A Vision-Language Model for UI and Infographics Understanding" (Google, 2024)
12. Lu et al., "OmniParser: Screen Parsing tool for Pure Vision Based GUI Agent" (Microsoft, 2024)

### Code Repositories

1. **timm**: https://github.com/huggingface/pytorch-image-models
2. **Swin Transformer**: https://github.com/microsoft/Swin-Transformer
3. **MAE**: https://github.com/facebookresearch/mae
4. **Deformable-DETR**: https://github.com/fundamentalvision/Deformable-DETR
5. **PyTorch Lightning**: https://github.com/Lightning-AI/pytorch-lightning
6. **DeepSpeed**: https://github.com/microsoft/DeepSpeed
7. **FFCV**: https://github.com/libffcv/ffcv
8. **WebDataset**: https://github.com/webdataset/webdataset

### Datasets

1. **Rico**: https://interactionmining.org/rico (72K Android screenshots)
2. **CLAY**: https://github.com/google-research-datasets/clay (iOS/Android layouts)
3. **Common Crawl**: https://commoncrawl.org/ (web screenshots)

---

## Next Steps

1. **Immediate (This Week)**:
   - Set up Lambda Labs account and provision 4× A100 instance
   - Implement basic synthetic UI generation
   - Train Swin-Small baseline on 10K examples
   - Validate full training pipeline

2. **Short-term (Weeks 2-4)**:
   - Generate 100K synthetic screenshots
   - Implement MAE pre-training
   - Train MAE on 100K unlabeled examples
   - Compare to training from scratch

3. **Medium-term (Weeks 5-8)**:
   - Scale to 1M examples
   - Implement full detection pipeline
   - Fine-tune and evaluate
   - Benchmark against baselines

4. **Long-term (Weeks 9-12)**:
   - Scale to 10M examples
   - Train foundation model
   - Comprehensive evaluation
   - Export for production use

---

**Document Status**: Initial research complete
**Next Review**: After Phase 1 POC (Week 2)
**Maintainer**: Joshua Spinak
