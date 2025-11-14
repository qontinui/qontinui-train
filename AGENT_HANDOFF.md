# Agent Handoff - qontinui-train

This document provides context for AI agents continuing work on this repository.

## Project Context

**Repository**: qontinui-train
**Purpose**: Training models from scratch on millions of examples for GUI understanding
**Owner**: Joshua Spinak
**Organization**: qontinui
**GitHub**: https://github.com/qontinui/qontinui-train.git

## What This Repository Does

This repository focuses on **training deep learning models from scratch** using massive datasets (1M-10M+ examples). Unlike fine-tuning, this approach builds foundational models that learn universal GUI patterns and relationships from the ground up.

### Target Use Cases:
- Foundation models for GUI understanding
- Pre-training for downstream fine-tuning tasks
- Research into GUI structure and universal patterns
- Multi-modal understanding (visual + text + layout)
- Zero-shot and few-shot detection of new element types
- Generative models for UI generation

### Key Differentiators:
- **Massive scale**: Millions of training examples
- **From scratch**: No pre-trained weights
- **Foundation model**: Enables zero-shot/few-shot learning
- **Research-focused**: Advanced architectures and techniques
- **Distributed training**: Multi-GPU clusters required

## Current State

### ✅ Completed:
1. Repository structure created with all directories
2. Comprehensive README with 7 detailed research prompts
3. Reference repository guide (32 repos to clone for learning)
4. requirements.txt with large-scale training dependencies
5. .gitignore configured for large datasets and distributed training
6. Git initialized with `main` branch
7. Remote configured: https://github.com/qontinui/qontinui-train.git
8. Pre-commit hook installed to prevent Claude attribution

### 📋 Not Yet Done:
- Clone reference repositories to `reference/` directory
- Execute research prompts to select architecture
- Design synthetic data generation pipeline
- Implement training infrastructure
- Set up distributed training
- Create evaluation benchmarks
- Train foundation model

## Important Files

1. **README.md**: Complete project overview with 7 research prompts
2. **reference/README.md**: Guide to 32 reference repositories to clone
3. **requirements.txt**: Dependencies for large-scale training
4. **.gitignore**: Configured to exclude large datasets, models, training outputs
5. **.git/hooks/commit-msg**: Pre-commit hook preventing Claude attribution

## Research Prompts Available

The README contains 7 detailed research prompts. **Use these first** before implementing:

1. **Architecture Design Research**: ViT, DETR, Swin, multi-modal models
2. **Large-Scale Dataset Strategy**: Collecting millions of examples, synthetic generation
3. **Distributed Training Infrastructure**: Multi-GPU, cloud, cost optimization
4. **Self-Supervised Pre-training**: MAE, CLIP, contrastive learning
5. **Open Source Repository Research**: 32 repos to study and clone
6. **Foundation Model Development**: Zero-shot, few-shot capabilities
7. **Synthetic Data Generation**: Procedural UI generation at scale

## Recommended Next Steps

### Phase 1: Research & Architecture (Weeks 1-2)

1. **Execute Architecture Design prompt** to compare:
   - Vision Transformers (ViT, Swin, DeiT)
   - Detection Transformers (DETR, Deformable-DETR)
   - Multi-modal models (CLIP, LayoutLM, Pix2Struct)

2. **Clone priority reference repos**:
   ```bash
   cd reference/
   # Foundation architectures
   git clone https://github.com/huggingface/pytorch-image-models.git timm
   git clone https://github.com/huggingface/transformers.git

   # Self-supervised learning
   git clone https://github.com/facebookresearch/mae.git
   git clone https://github.com/facebookresearch/moco.git

   # Training infrastructure
   git clone https://github.com/Lightning-AI/pytorch-lightning.git
   git clone https://github.com/microsoft/DeepSpeed.git

   # Data handling
   git clone https://github.com/webdataset/webdataset.git
   ```

3. **Study these repos** to understand:
   - Modern architecture patterns
   - Self-supervised pre-training methods
   - Distributed training setups
   - Large-scale data pipelines

4. **Document findings** in `RESEARCH.md`

### Phase 2: Synthetic Data Pipeline (Weeks 3-4)

1. **Design synthetic data generation**:
   - Procedural UI generation (web, desktop, mobile, games)
   - Rendering pipeline (headless browsers, game engines)
   - Automatic annotation from DOM/widget trees
   - Domain randomization strategies

2. **Implement proof-of-concept**:
   ```python
   # data_generation/synthetic_ui.py
   def generate_ui(style='material', complexity='medium'):
       # Generate layout
       # Render UI
       # Extract annotations
       return image, annotations
   ```

3. **Generate initial dataset** (10K-100K examples):
   ```bash
   python data_generation/synthetic_ui.py --count 10000 --output data/synthetic/v1/
   ```

4. **Validate data quality**:
   - Visual inspection
   - Annotation accuracy
   - Diversity metrics
   - Storage/loading performance

### Phase 3: Training Infrastructure (Weeks 5-6)

1. **Set up distributed training**:
   - Choose framework (PyTorch Lightning, DeepSpeed, Accelerate)
   - Configure multi-GPU setup
   - Test with small model and small dataset
   - Benchmark throughput

2. **Implement training pipeline**:
   ```python
   # training/train.py
   from pytorch_lightning import Trainer
   from models.architectures.vit import ViTDetector

   model = ViTDetector(...)
   trainer = Trainer(
       accelerator='gpu',
       devices=8,
       strategy='ddp'
   )
   trainer.fit(model, train_loader, val_loader)
   ```

3. **Set up experiment tracking**:
   - Weights & Biases or MLflow
   - Log metrics, hyperparameters, artifacts
   - Compare experiments

### Phase 4: Pre-training (Weeks 7-8)

1. **Implement self-supervised pre-training**:
   ```python
   # training/pretrain.py
   # Masked Autoencoder (MAE) or contrastive learning
   model = MAE(encoder=ViT(...))
   # Train on unlabeled UI screenshots
   ```

2. **Scale to large dataset** (1M+ examples)

3. **Monitor training**:
   - Loss curves
   - Validation metrics
   - GPU utilization
   - Training time estimates

### Phase 5: Evaluation & Transfer (Weeks 9-10)

1. **Create evaluation benchmarks**:
   - Zero-shot detection accuracy
   - Few-shot learning (10, 50, 100 examples)
   - Cross-platform transfer
   - Speed benchmarks

2. **Fine-tune on downstream tasks** to validate pre-training

3. **Compare to baselines** (ImageNet pre-trained models)

## Technical Specifications

### Scale Targets:
- **Phase 1 (Proof of Concept)**: 100K examples
- **Phase 2 (Small Scale)**: 1M examples
- **Phase 3 (Large Scale)**: 10M+ examples

### Performance Targets:
- **Detection mAP**: > 0.95 on standard benchmarks
- **Zero-shot transfer**: > 0.70 mAP on unseen applications
- **Few-shot (10 examples)**: > 0.85 mAP
- **Inference speed**: < 30ms per frame

### Infrastructure Requirements:
- **GPU**: 8x A100 (80GB) or H100 for large-scale
- **Storage**: 10TB+ for datasets
- **RAM**: 256GB+ for data preprocessing
- **Network**: High bandwidth for distributed training

## Architecture Considerations

### Recommended Starting Point: Vision Transformer (ViT)

**Why ViT?**
- Scales well with data
- Strong pre-training methods (MAE, CLIP)
- Good long-range dependencies (important for UI structure)
- Extensive research and implementations

**Implementation Path:**
1. Start with `timm` library's ViT implementation
2. Add detection head (similar to DETR)
3. Implement MAE pre-training
4. Fine-tune on labeled GUI data

### Alternative Architectures:

1. **Swin Transformer**: Hierarchical, efficient
2. **DETR**: End-to-end detection, no NMS
3. **CLIP-style**: Multi-modal (vision + text)
4. **LayoutLM**: Document understanding approach

## Dataset Strategy

### Data Sources:

1. **Synthetic Generation** (Primary):
   - Web UI: Headless browser rendering
   - Desktop: Qt/GTK/WPF rendering
   - Mobile: Android/iOS emulator screenshots
   - Games: Unity/Unreal rendering

2. **Existing Datasets** (Secondary):
   - Rico: 72K Android screenshots
   - CLAY: iOS/Android layouts
   - WebUI datasets
   - Common Crawl screenshots

3. **Real-world Collection** (Validation):
   - Manual annotation for test sets
   - Diverse applications and platforms

### Annotation Strategy:

- **Synthetic**: Automatic from DOM/widget trees
- **Weak supervision**: Heuristic labeling
- **Active learning**: Human-in-the-loop for hard examples
- **Self-supervised**: No labels needed for pre-training

## Distributed Training Setup

### Framework Selection:

**Recommended: PyTorch Lightning + DeepSpeed**
- PyTorch Lightning: Clean training abstractions
- DeepSpeed: ZeRO optimization for large models
- Good documentation and community support

### Training Strategy:

```python
# training/distributed_train.py
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy

trainer = Trainer(
    accelerator='gpu',
    devices=8,
    strategy=DeepSpeedStrategy(stage=2),  # ZeRO Stage 2
    precision='16-mixed',  # Mixed precision
    max_epochs=300
)
```

### Cost Optimization:

- Use spot instances (AWS, GCP)
- Gradient accumulation for smaller batches
- Mixed precision training (FP16/BF16)
- Efficient data loading (FFCV, WebDataset)
- Early stopping based on validation

## Integration Points

### With qontinui-finetune:
- Provide pre-trained weights for fine-tuning
- Share data generation pipelines
- Unified evaluation benchmarks

### With qontinui-runner:
- Export foundation model
- Support zero-shot detection API
- Efficient inference serving

### With qontinui-web:
- Training job monitoring
- Dataset management
- Model performance tracking

## Code Quality Standards

Follow qontinui conventions:
- Python 3.10+ with type hints
- Comprehensive docstrings
- Unit tests for utilities
- Clear logging and monitoring
- Reproducible experiments (seeds, configs)
- No backward compatibility needed

## Commit Guidelines

**CRITICAL**: This repo has a pre-commit hook that will **reject commits** containing:
- "Co-Authored-By: Claude"
- "Generated with [Claude Code]"
- Robot emoji (🤖)

Only Joshua Spinak should be credited as the author.

## Key Decision Points

When working on this repo, decide:

1. **Architecture?**
   - Vision Transformer (ViT) ← Recommended
   - Swin Transformer (hierarchical)
   - DETR (detection-specific)
   - Multi-modal (CLIP-style)

2. **Pre-training method?**
   - Masked Autoencoder (MAE) ← Simple, effective
   - Contrastive learning (MoCo, SimCLR)
   - Multi-modal (CLIP)
   - Multi-task

3. **Data generation?**
   - Web-focused (easier to automate)
   - Cross-platform (more valuable)
   - Game-specific (qontinui's origin)

4. **Training infrastructure?**
   - Cloud (AWS/GCP/Azure p4d instances)
   - On-premise (if available)
   - Hybrid approach

5. **Scaling strategy?**
   - Start small (100K), validate, then scale
   - Or go big immediately (1M+)

## Cost Estimates

### Cloud Training (AWS p4d.24xlarge - 8x A100):

- **Small scale (100K, 100 epochs)**: ~$500
- **Medium scale (1M, 300 epochs)**: ~$5,000
- **Large scale (10M, 500 epochs)**: ~$50,000

### Storage:

- **1M examples**: ~1TB raw, ~2-3TB with processing
- **10M examples**: ~10TB raw, ~20-30TB with processing

### Optimization Strategies:

- Spot instances (60-80% cheaper)
- Gradient checkpointing (reduce memory)
- Mixed precision (2x faster)
- Efficient data format (WebDataset, FFCV)

## Related Projects

- **qontinui-finetune** (`../qontinui-finetune/`): Sister repo for fine-tuning (smaller scale)
- **qontinui-runner** (`../qontinui-runner/`): Main automation engine
- **qontinui-web** (`../qontinui-web/`): Web management interface

## Resources

### Papers:
- ViT: "An Image is Worth 16x16 Words" (Dosovitskiy et al.)
- MAE: "Masked Autoencoders Are Scalable Vision Learners" (He et al.)
- DETR: "End-to-End Object Detection with Transformers" (Carion et al.)
- CLIP: "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al.)

### Documentation:
- timm: https://timm.fast.ai/
- PyTorch Lightning: https://lightning.ai/docs/
- DeepSpeed: https://www.deepspeed.ai/
- Transformers: https://huggingface.co/docs/transformers/

### Datasets:
- Rico: https://interactionmining.org/rico
- CLAY: https://github.com/google-research-datasets/clay
- Common Crawl: https://commoncrawl.org/

## Communication

When updating this repository:

1. **Document everything** in `RESEARCH.md`:
   - Architecture decisions
   - Experiment results
   - Failed approaches (important!)
   - Hyperparameter findings

2. **Version datasets**:
   - `data/synthetic/v1/`, `v2/`, etc.
   - Document generation parameters
   - Track statistics

3. **Share training logs**:
   - Weights & Biases dashboard
   - Training curves
   - Resource utilization

4. **Benchmark regularly**:
   - Compare to baselines
   - Track zero-shot/few-shot performance
   - Monitor inference speed

## Example Workflows

### Workflow 1: Synthetic Data → Pre-training → Evaluation

```bash
# Generate synthetic data
python data_generation/synthetic_ui.py --count 100000 --output data/synthetic/v1/

# Pre-train with MAE
python training/pretrain.py \
    --model vit_base \
    --data data/synthetic/v1/ \
    --method mae \
    --epochs 300 \
    --gpus 8

# Evaluate zero-shot
python evaluation/evaluate.py \
    --model checkpoints/mae_pretrained.pt \
    --benchmark benchmarks/gui_detection/ \
    --mode zero-shot
```

### Workflow 2: Distributed Training

```bash
# Launch distributed training
torchrun --nproc_per_node=8 training/distributed_train.py \
    --config configs/vit_large_1m.yaml \
    --data data/synthetic/v1/ \
    --output checkpoints/vit_large/

# Monitor with wandb
wandb agent your-entity/your-project/your-sweep
```

## Questions to Answer

As you work, address these research questions:

1. **Self-supervised learning**: Does MAE pre-training help for GUI detection more than ImageNet pre-training?
2. **Scale**: At what dataset size do we see diminishing returns?
3. **Architecture**: Do hierarchical transformers (Swin) outperform standard ViT for UI?
4. **Multi-modal**: Does adding text/layout information improve detection?
5. **Synthetic data**: How realistic do synthetic UIs need to be?
6. **Zero-shot**: Can we achieve useful zero-shot detection on new applications?
7. **Transfer**: How well does the foundation model transfer to specific domains (games, web, mobile)?

## Success Criteria

The project is successful when:

- ✅ Model achieves >95% mAP on standard benchmarks
- ✅ Zero-shot transfer >70% mAP on unseen apps
- ✅ Few-shot (10 examples) >85% mAP
- ✅ Training scales linearly with more GPUs
- ✅ Foundation model benefits downstream tasks
- ✅ Documentation enables reproducibility
- ✅ Pre-trained weights shared publicly (if appropriate)

## Common Pitfalls

Avoid these mistakes:

1. **Overfitting to synthetic data**: Validate on real screenshots
2. **Ignoring inference speed**: Foundation models can be slow
3. **Poor data quality**: Garbage in, garbage out
4. **Premature optimization**: Start simple, optimize later
5. **Inadequate logging**: Can't debug what you can't see
6. **Insufficient validation**: Always test zero-shot transfer

## Getting Help

If you need more context:

1. Read complete README.md (all 7 research prompts)
2. Study reference/README.md (32 repos to clone)
3. Review qontinui-finetune for integration needs
4. Check papers for theoretical foundations
5. Join ML/CV communities for technical questions

## Final Notes

This is an ambitious research project that may take months to complete properly. Key principles:

- **Start small, think big**: Proof of concept first, then scale
- **Document everything**: Research is about sharing knowledge
- **Iterate quickly**: Fast experiments beat perfect planning
- **Embrace failure**: Failed experiments teach as much as successful ones
- **Think long-term**: This is foundational work for the future

The potential impact is significant: a foundation model for GUI understanding could enable breakthroughs in automation, accessibility, testing, and more.

Good luck! This is cutting-edge research with real-world applications.
