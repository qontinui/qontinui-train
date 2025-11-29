# Reference Repositories for Training from Scratch

This directory contains cloned reference repositories for large-scale model training from scratch.

## Recommended Repositories to Clone

### Vision Transformer Architectures

1. **timm (PyTorch Image Models)**
   - URL: https://github.com/huggingface/pytorch-image-models
   - Purpose: Comprehensive vision model library with training scripts
   - Key features: ViT, Swin, ConvNeXt, training recipes
   - Clone command: `git clone https://github.com/huggingface/pytorch-image-models.git`

2. **transformers (Hugging Face)**
   - URL: https://github.com/huggingface/transformers
   - Purpose: Transformer models including vision transformers
   - Key features: ViT, DeiT, BEiT, Swin, pre-training examples
   - Clone command: `git clone https://github.com/huggingface/transformers.git`

3. **Swin-Transformer**
   - URL: https://github.com/microsoft/Swin-Transformer
   - Purpose: Official Swin Transformer implementation
   - Key features: Hierarchical vision transformer, training from scratch
   - Clone command: `git clone https://github.com/microsoft/Swin-Transformer.git`

4. **DeiT (Data-efficient Image Transformers)**
   - URL: https://github.com/facebookresearch/deit
   - Purpose: Efficient ViT training
   - Key features: Distillation, efficient training recipes
   - Clone command: `git clone https://github.com/facebookresearch/deit.git`

### Detection Transformers

5. **DETR (Detection Transformer)**
   - URL: https://github.com/facebookresearch/detr
   - Purpose: End-to-end object detection with transformers
   - Key features: No NMS, query-based detection
   - Clone command: `git clone https://github.com/facebookresearch/detr.git`

6. **Deformable-DETR**
   - URL: https://github.com/fundamentalvision/Deformable-DETR
   - Purpose: Improved DETR with deformable attention
   - Key features: Faster convergence, multi-scale
   - Clone command: `git clone https://github.com/fundamentalvision/Deformable-DETR.git`

### Self-Supervised Learning

7. **MAE (Masked Autoencoders)**
   - URL: https://github.com/facebookresearch/mae
   - Purpose: Self-supervised pre-training via masked image modeling
   - Key features: Simple, effective pre-training
   - Clone command: `git clone https://github.com/facebookresearch/mae.git`

8. **MoCo (Momentum Contrast)**
   - URL: https://github.com/facebookresearch/moco
   - Purpose: Contrastive learning for visual representations
   - Key features: Momentum encoder, large queues
   - Clone command: `git clone https://github.com/facebookresearch/moco.git`

9. **SimCLR**
   - URL: https://github.com/google-research/simclr
   - Purpose: Simple framework for contrastive learning
   - Key features: Data augmentation strategies
   - Clone command: `git clone https://github.com/google-research/simclr.git`

10. **BEiT**
    - URL: https://github.com/microsoft/unilm/tree/master/beit
    - Purpose: BERT pre-training for images
    - Key features: Masked image modeling with discrete tokens
    - Clone command: `git clone https://github.com/microsoft/unilm.git` (navigate to beit/)

11. **lightly**
    - URL: https://github.com/lightly-ai/lightly
    - Purpose: Self-supervised learning library
    - Key features: Multiple methods, easy to use
    - Clone command: `git clone https://github.com/lightly-ai/lightly.git`

### Multi-modal Learning

12. **CLIP**
    - URL: https://github.com/openai/CLIP
    - Purpose: Vision-language contrastive learning
    - Key features: Zero-shot transfer, text-image alignment
    - Clone command: `git clone https://github.com/openai/CLIP.git`

13. **OpenCLIP**
    - URL: https://github.com/mlfoundations/open_clip
    - Purpose: Open source CLIP training
    - Key features: Training from scratch, multiple architectures
    - Clone command: `git clone https://github.com/mlfoundations/open_clip.git`

14. **LayoutLM**
    - URL: https://github.com/microsoft/unilm/tree/master/layoutlm
    - Purpose: Document understanding with layout
    - Key features: Multi-modal pre-training (text + layout + visual)
    - Clone command: Clone unilm repo, navigate to layoutlm/

15. **Pix2Struct**
    - URL: https://github.com/google-research/pix2struct
    - Purpose: Screenshot understanding
    - Key features: Designed for UI/document understanding
    - Clone command: `git clone https://github.com/google-research/pix2struct.git`

### Distributed Training Infrastructure

16. **PyTorch Lightning**
    - URL: https://github.com/Lightning-AI/pytorch-lightning
    - Purpose: Simplified distributed training
    - Key features: Multi-GPU, TPU support, clean abstractions
    - Clone command: `git clone https://github.com/Lightning-AI/pytorch-lightning.git`

17. **DeepSpeed**
    - URL: https://github.com/microsoft/DeepSpeed
    - Purpose: Large-scale model training optimization
    - Key features: ZeRO, pipeline parallelism, efficiency
    - Clone command: `git clone https://github.com/microsoft/DeepSpeed.git`

18. **FairScale**
    - URL: https://github.com/facebookresearch/fairscale
    - Purpose: PyTorch extensions for large-scale training
    - Key features: FSDP, pipeline parallelism
    - Clone command: `git clone https://github.com/facebookresearch/fairscale.git`

19. **Composer (MosaicML)**
    - URL: https://github.com/mosaicml/composer
    - Purpose: Efficient training with algorithmic speedups
    - Key features: Training recipes, efficiency techniques
    - Clone command: `git clone https://github.com/mosaicml/composer.git`

20. **Accelerate (Hugging Face)**
    - URL: https://github.com/huggingface/accelerate
    - Purpose: Simple distributed training
    - Key features: Device-agnostic code, mixed precision
    - Clone command: `git clone https://github.com/huggingface/accelerate.git`

### Data Loading and Management

21. **WebDataset**
    - URL: https://github.com/webdataset/webdataset
    - Purpose: Efficient large-scale dataset handling
    - Key features: Tar-based storage, streaming
    - Clone command: `git clone https://github.com/webdataset/webdataset.git`

22. **FFCV**
    - URL: https://github.com/libffcv/ffcv
    - Purpose: Fast data loading library
    - Key features: Ultra-fast ImageNet training
    - Clone command: `git clone https://github.com/libffcv/ffcv.git`

23. **NVIDIA DALI**
    - URL: https://github.com/NVIDIA/DALI
    - Purpose: GPU-accelerated data loading
    - Key features: Pipeline on GPU, fast augmentation
    - Clone command: `git clone https://github.com/NVIDIA/DALI.git`

### Experiment Tracking

24. **Weights & Biases (wandb)**
    - URL: https://github.com/wandb/wandb
    - Purpose: Experiment tracking and visualization
    - Key features: Logging, sweeps, artifacts
    - Clone command: `git clone https://github.com/wandb/wandb.git`

25. **MLflow**
    - URL: https://github.com/mlflow/mlflow
    - Purpose: ML lifecycle management
    - Key features: Tracking, projects, models, registry
    - Clone command: `git clone https://github.com/mlflow/mlflow.git`

### Synthetic Data Generation

26. **Unity ML-Agents**
    - URL: https://github.com/Unity-Technologies/ml-agents
    - Purpose: Unity-based synthetic data generation
    - Key features: Procedural generation, rendering
    - Clone command: `git clone https://github.com/Unity-Technologies/ml-agents.git`

27. **UnrealCV**
    - URL: https://github.com/unrealcv/unrealcv
    - Purpose: Unreal Engine for computer vision
    - Key features: Photorealistic rendering, annotations
    - Clone command: `git clone https://github.com/unrealcv/unrealcv.git`

### Complete Training Frameworks

28. **MMDetection**
    - URL: https://github.com/open-mmlab/mmdetection
    - Purpose: Comprehensive detection framework
    - Key features: Many architectures, training from scratch
    - Clone command: `git clone https://github.com/open-mmlab/mmdetection.git`

29. **Detectron2**
    - URL: https://github.com/facebookresearch/detectron2
    - Purpose: Facebook's detection platform
    - Key features: Training from scratch support
    - Clone command: `git clone https://github.com/facebookresearch/detectron2.git`

### UI-Specific Research

30. **Rico Dataset**
    - URL: https://github.com/google-research-datasets/rico
    - Purpose: Large-scale mobile UI dataset
    - Key features: 72k+ screens with annotations
    - Clone command: `git clone https://github.com/google-research-datasets/rico.git`

31. **Screen2Vec**
    - URL: https://github.com/google-research/google-research/tree/master/screen2vec
    - Purpose: Learning embeddings of UI screens
    - Key features: Self-supervised screen understanding
    - Clone command: Clone google-research, navigate to screen2vec/

32. **Widget Captioning**
    - URL: https://github.com/google-research/google-research/tree/master/widget_caption
    - Purpose: Generating descriptions for UI elements
    - Key features: Multi-modal UI understanding
    - Clone command: Clone google-research, navigate to widget_caption/

## How to Use These References

1. **Clone to this directory**:
   ```bash
   cd reference/
   git clone <repository-url>
   ```

2. **Study training pipelines**:
   - Data preprocessing and loading
   - Model architecture implementations
   - Training loops and optimization
   - Distributed training setup
   - Evaluation and checkpointing

3. **Learn training recipes**:
   - Hyperparameter configurations
   - Learning rate schedules
   - Augmentation strategies
   - Pre-training objectives
   - Transfer learning approaches

4. **Adapt for large-scale GUI training**:
   - Modify data loaders for GUI datasets
   - Customize architectures for UI patterns
   - Implement GUI-specific pre-training tasks
   - Scale to millions of examples

5. **Extract infrastructure code**:
   - Distributed training utilities
   - Efficient data loading
   - Experiment tracking
   - Model checkpointing
   - Evaluation metrics

## Priority Order for Study

### Phase 1: Foundation
1. timm - Learn modern architectures
2. PyTorch Lightning - Training infrastructure
3. MAE - Self-supervised learning
4. WebDataset - Large-scale data handling

### Phase 2: Architecture Exploration
5. Swin Transformer - Hierarchical ViTs
6. DETR - Detection transformers
7. CLIP - Multi-modal learning
8. Pix2Struct - UI-specific models

### Phase 3: Scaling
9. DeepSpeed - Distributed training
10. FFCV - Fast data loading
11. Composer - Training efficiency
12. Rico - UI dataset handling

## Git Ignore

Add to `.gitignore`:
```
reference/*/
!reference/README.md
```

## License Tracking

Maintain a `LICENSES.md` file tracking:
- Repository name and URL
- License type (MIT, Apache 2.0, etc.)
- Usage in qontinui-train
- Attribution requirements

## Maintenance

- Keep reference repos updated
- Document code adaptations in RESEARCH.md
- Track versions used for reproducibility
- Test compatibility with new releases
