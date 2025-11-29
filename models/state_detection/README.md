# State Detection Models

This directory contains models for ML-based state detection in GUI automation. State detection analyzes sequences of screenshots to identify distinct application states and predict transitions between them.

## What is State Detection?

State detection is a higher-level task compared to element detection:

- **Element Detection**: Identifies individual UI components (buttons, text fields, icons) within a single screenshot
- **State Detection**: Identifies the overall application state and predicts how states change over time based on screenshot sequences

### Key Differences

| Aspect | Element Detection | State Detection |
|--------|------------------|-----------------|
| Input | Single screenshot | Sequence of screenshots |
| Output | Bounding boxes + labels for UI elements | State labels + transition predictions |
| Focus | Individual components | Overall application state |
| Temporal | No temporal modeling | Models state evolution over time |
| Granularity | Fine-grained (pixels/components) | Coarse-grained (application-level) |

## Models

### RegionProposalNetwork

Proposes candidate state regions from screenshots. These regions represent coherent areas that define application states.

**Input**: Feature maps from backbone CNN/ViT `[B, C, H, W]`

**Output**:
- Proposal boxes `[N, 4]` in (x1, y1, x2, y2) format
- Confidence scores `[N]` for each proposal

**Use Cases**:
- Identifying login screen regions vs. main application regions
- Detecting popup dialogs that represent temporary states
- Proposing menu areas that define navigation states
- Distinguishing between loading states and content states

**Example**:
```python
from models.state_detection import RegionProposalNetwork

# Initialize RPN
rpn = RegionProposalNetwork(
    backbone_dim=768,
    num_anchors=9,
    proposal_count=100
)

# Generate proposals from features
proposals, scores = rpn(feature_maps)
```

### TransitionPredictor

Predicts state transitions based on sequences of screenshots. Uses sequential modeling (LSTM + attention) to understand temporal patterns.

**Input**:
- Screenshot sequence features `[B, T, D]` where T is sequence length
- Optional: Current state labels `[B, T]`

**Output**:
- Next state predictions `[B, num_states]`
- Learned transition probability matrix `[num_states, num_states]`
- Attention weights showing which frames are most relevant `[B, T, T]`

**Use Cases**:
- Predicting that clicking a login button transitions to main screen
- Learning that error dialogs typically return to previous state
- Identifying cyclical patterns in navigation flows
- Detecting unexpected transitions that may indicate errors
- Understanding action-state relationships

**Example**:
```python
from models.state_detection import TransitionPredictor

# Initialize predictor
predictor = TransitionPredictor(
    feature_dim=768,
    hidden_dim=512,
    num_states=10,
    sequence_length=5
)

# Predict next state
output = predictor(sequence_features)
next_state = output['predictions']
transition_matrix = output['transition_probs']
```

## Training Data Requirements

State detection models require:

1. **Screenshot Sequences**: Series of screenshots showing GUI state changes
2. **State Labels**: Labels for each screenshot indicating the application state
3. **Transition Annotations**: Information about what action caused transitions (optional but helpful)
4. **Temporal Alignment**: Timestamps to ensure proper sequence ordering

See `data/screenshot_sequences/` for data format specifications.

## Architecture Overview

```
Screenshot Sequence
       |
       v
   Backbone (ViT/ResNet)
       |
       +-- Region Proposal Network --> State Region Proposals
       |
       v
   Feature Sequence [T x D]
       |
       v
   Transition Predictor (LSTM + Attention)
       |
       v
   State Predictions + Transition Matrix
```

## Integration with Element Detection

State detection complements element detection:

1. **Element detection** identifies what UI components are present
2. **State detection** determines what state the application is in
3. Together, they provide a complete understanding of the GUI

For example:
- Element detection finds: "login button", "username field", "password field"
- State detection identifies: "Login Screen State"
- Combined understanding: "We are on the login screen with visible login components"

## Future Enhancements

- [ ] Multi-modal state detection (screenshots + DOM/accessibility tree)
- [ ] Hierarchical state modeling (sub-states within states)
- [ ] Action-conditioned transition prediction
- [ ] Uncertainty estimation for state predictions
- [ ] Few-shot state detection for new applications

## References

- Region Proposal Networks: Faster R-CNN paper
- Sequential modeling: LSTM and Transformer architectures
- Temporal action detection: Video understanding literature
