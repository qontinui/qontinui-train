# Screenshot Sequences for State Detection Training

This directory contains screenshot sequences used for training state detection models. Unlike single-image datasets for element detection, state detection requires temporal sequences to learn state transitions.

## Data Format

### Directory Structure

```
screenshot_sequences/
├── app_name_1/
│   ├── sequence_001/
│   │   ├── metadata.json
│   │   ├── frames/
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   └── ...
│   │   └── annotations.json
│   ├── sequence_002/
│   │   └── ...
│   └── ...
├── app_name_2/
│   └── ...
└── README.md
```

### Metadata Format (metadata.json)

Each sequence should have a metadata file describing the sequence:

```json
{
  "sequence_id": "app_name_1_seq_001",
  "app_name": "Example App",
  "app_version": "1.2.3",
  "platform": "windows",
  "resolution": [1920, 1080],
  "fps": 2.0,
  "duration_seconds": 30.0,
  "frame_count": 60,
  "capture_date": "2024-01-15T10:30:00Z",
  "description": "User navigating from login to dashboard",
  "initial_state": "login_screen",
  "final_state": "dashboard",
  "num_transitions": 3
}
```

### Annotations Format (annotations.json)

Each sequence should have detailed annotations:

```json
{
  "sequence_id": "app_name_1_seq_001",
  "frames": [
    {
      "frame_id": 0,
      "timestamp": 0.0,
      "state_label": "login_screen",
      "state_id": 0,
      "state_regions": [
        {
          "bbox": [100, 200, 800, 600],
          "region_type": "form",
          "confidence": 1.0
        }
      ],
      "actions": []
    },
    {
      "frame_id": 1,
      "timestamp": 0.5,
      "state_label": "login_screen",
      "state_id": 0,
      "state_regions": [
        {
          "bbox": [100, 200, 800, 600],
          "region_type": "form",
          "confidence": 1.0
        }
      ],
      "actions": [
        {
          "type": "click",
          "target": "login_button",
          "coordinates": [450, 500]
        }
      ]
    },
    {
      "frame_id": 2,
      "timestamp": 1.0,
      "state_label": "loading",
      "state_id": 1,
      "state_regions": [
        {
          "bbox": [400, 300, 1520, 780],
          "region_type": "loading_indicator",
          "confidence": 0.95
        }
      ],
      "actions": []
    },
    {
      "frame_id": 3,
      "timestamp": 1.5,
      "state_label": "dashboard",
      "state_id": 2,
      "state_regions": [
        {
          "bbox": [0, 0, 1920, 1080],
          "region_type": "full_screen",
          "confidence": 1.0
        }
      ],
      "actions": []
    }
  ],
  "state_labels": {
    "0": "login_screen",
    "1": "loading",
    "2": "dashboard",
    "3": "settings",
    "4": "error_dialog"
  },
  "transitions": [
    {
      "from_frame": 1,
      "to_frame": 2,
      "from_state": 0,
      "to_state": 1,
      "trigger_action": "click_login",
      "transition_type": "user_initiated"
    },
    {
      "from_frame": 2,
      "to_frame": 3,
      "from_state": 1,
      "to_state": 2,
      "trigger_action": "auto",
      "transition_type": "automatic"
    }
  ]
}
```

## Data Requirements

### Minimum Requirements for Training

1. **Sequence Length**: At least 5-10 frames per sequence to capture temporal patterns
2. **Number of Sequences**: Minimum 100 sequences per application for basic training
3. **State Coverage**: At least 5-10 distinct states per application
4. **Transition Coverage**: Each state pair should have at least 5 example transitions
5. **Resolution**: Consistent resolution within each application (can vary across apps)

### Recommended Best Practices

1. **Temporal Sampling**: Capture at 1-5 FPS (faster for quick transitions, slower for stable states)
2. **State Balance**: Ensure relatively balanced representation of different states
3. **Transition Diversity**: Include different ways to reach the same state
4. **Edge Cases**: Include error states, timeouts, and unexpected behaviors
5. **Natural Interactions**: Capture realistic user interaction patterns

## Differences from Element Detection Data

| Aspect | Element Detection | State Detection |
|--------|------------------|-----------------|
| **Unit** | Single screenshot | Sequence of screenshots |
| **Labels** | Bounding boxes + element types | State labels + transitions |
| **Temporal** | No temporal information | Timestamps and frame order crucial |
| **Annotations** | Per-image element labels | Per-frame state + sequence-level transitions |
| **Size** | ~1000s of images | ~100s of sequences (but 1000s of frames) |

## Expected Inputs for Training

The training pipeline expects:

1. **Screenshot Sequences**:
   - Format: PNG or JPG
   - Resolution: Any (will be resized), but consistent within app
   - Color space: RGB
   - Naming: Sequential (000000.png, 000001.png, etc.)

2. **State Labels**:
   - Per-frame state labels (string or integer)
   - State vocabulary (mapping of state IDs to names)
   - Hierarchical states (optional, for nested state modeling)

3. **Transition Information**:
   - Source and target frames
   - Source and target states
   - Triggering action (if known)
   - Transition type (user-initiated, automatic, error-triggered)

4. **State Region Annotations** (optional but recommended):
   - Bounding boxes for state-defining regions
   - Region types (form, dialog, menu, content area, etc.)
   - Confidence scores

## Expected Outputs from Models

Trained models will produce:

1. **State Predictions**: Classification of each frame's state
2. **Transition Probabilities**: Learned transition matrix between states
3. **State Regions**: Proposed bounding boxes for state-defining regions
4. **Attention Weights**: Which frames in a sequence are most informative
5. **Confidence Scores**: Uncertainty estimates for predictions

## Example Use Cases

### Use Case 1: Login Flow Detection
```
Sequence: login_screen → (enter credentials) → login_screen →
          (click login) → loading → dashboard

Goal: Learn that login button click triggers transition to dashboard
```

### Use Case 2: Error State Detection
```
Sequence: form_entry → (submit invalid) → error_dialog →
          (dismiss) → form_entry

Goal: Learn that errors are temporary states that return to previous state
```

### Use Case 3: Navigation Pattern Learning
```
Sequence: home → menu → settings → menu → profile → menu → home

Goal: Learn that menu is a hub state with transitions to multiple destinations
```

## Data Collection Tips

1. **Record Real User Sessions**: Capture actual user interactions for realistic patterns
2. **Script Common Flows**: Automate collection of standard interaction sequences
3. **Include Edge Cases**: Deliberately trigger errors, timeouts, and edge conditions
4. **Vary Interaction Speed**: Capture both fast and slow interaction patterns
5. **Document Context**: Record what actions triggered each transition

## Data Augmentation

For screenshot sequences, consider:

1. **Temporal Augmentation**:
   - Frame dropping (simulate lower FPS)
   - Frame interpolation (simulate higher FPS)
   - Subsequence extraction
   - Reverse sequences (for symmetric interactions)

2. **Visual Augmentation** (per-frame):
   - Color jittering
   - Brightness/contrast adjustment
   - Random cropping (maintaining state regions)
   - Resolution changes

3. **Sequence-Level Augmentation**:
   - Concatenate related sequences
   - Add synthetic pauses (duplicate frames)
   - Insert noise frames

## Validation and Testing

Split data by:

1. **Sequence-Level Split**: Entire sequences go to train/val/test (prevents leakage)
2. **State Coverage**: Ensure all states appear in train/val/test
3. **Transition Coverage**: Ensure all transition types are represented
4. **Application Diversity**: If multi-app, ensure apps split across sets

Recommended split: 70% train / 15% validation / 15% test

## Future Enhancements

- [ ] Support for video files (instead of frame sequences)
- [ ] Automatic state detection from unlabeled sequences
- [ ] Multi-modal data (screenshots + DOM + accessibility tree)
- [ ] Hierarchical state annotations
- [ ] Action-state causality annotations
- [ ] Cross-application state alignment
