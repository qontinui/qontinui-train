"""
Transition Predictor for State Detection

This module implements a model for predicting state transitions based on sequences of
screenshots. It analyzes temporal patterns to understand how GUI states change over time.

The transition predictor is designed to:
1. Process sequences of screenshots with their state labels
2. Learn temporal patterns in state transitions
3. Predict the next likely state given a current sequence
4. Identify transition triggers (actions that cause state changes)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class TransitionPredictor(nn.Module):
    """
    Model for predicting state transitions in GUI sequences.

    This model differs from element detection by focusing on temporal dynamics and
    state-level changes rather than individual UI component detection. It uses
    sequential modeling to understand patterns in how states evolve.

    Args:
        feature_dim (int): Dimension of input features from backbone
        hidden_dim (int): Dimension of hidden state in sequential model
        num_states (int): Number of possible states to predict
        sequence_length (int): Length of input sequences
        dropout (float): Dropout rate for regularization

    Input:
        sequence_features (torch.Tensor): Features from screenshot sequence [B, T, D]
        state_labels (torch.Tensor, optional): Ground truth state labels [B, T]

    Output:
        predictions (torch.Tensor): Predicted next state probabilities [B, num_states]
        transition_matrix (torch.Tensor): Learned transition probabilities [num_states, num_states]

    Example use cases:
        - Predicting that clicking a login button transitions to main screen
        - Learning that error dialogs typically return to previous state
        - Identifying cyclical patterns in navigation flows
        - Detecting unexpected state transitions that may indicate errors
    """

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 512,
        num_states: int = 10,
        sequence_length: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_states = num_states
        self.sequence_length = sequence_length

        # Sequential encoder (LSTM for temporal modeling)
        self.encoder = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        # Attention mechanism for focusing on relevant frames
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, num_heads=8, dropout=dropout, batch_first=True
        )

        # State classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_states),
        )

        # Transition matrix (learnable transition probabilities)
        self.transition_matrix = nn.Parameter(torch.randn(num_states, num_states))

    def forward(
        self,
        sequence_features: torch.Tensor,
        state_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to predict state transitions.

        Args:
            sequence_features: Features from screenshot sequence [B, T, D]
            state_labels: Optional ground truth labels [B, T]

        Returns:
            Dictionary containing:
                - 'predictions': Next state probabilities [B, num_states]
                - 'sequence_predictions': State predictions for each frame [B, T, num_states]
                - 'transition_probs': Normalized transition matrix [num_states, num_states]
                - 'attention_weights': Attention weights over sequence [B, T, T]
        """
        batch_size, seq_len, _ = sequence_features.shape

        # Encode sequence with LSTM
        encoded, (hidden, cell) = self.encoder(sequence_features)

        # Apply self-attention to focus on relevant frames
        attended, attention_weights = self.attention(encoded, encoded, encoded)

        # Generate predictions for each frame
        sequence_predictions = self.classifier(attended)

        # Predict next state (using last frame's representation)
        next_state_pred = sequence_predictions[:, -1, :]

        # Normalize transition matrix to get probabilities
        transition_probs = torch.softmax(self.transition_matrix, dim=1)

        return {
            "predictions": next_state_pred,
            "sequence_predictions": sequence_predictions,
            "transition_probs": transition_probs,
            "attention_weights": attention_weights,
        }

    def predict_transition(
        self, current_state: int, sequence_features: torch.Tensor
    ) -> Tuple[int, float]:
        """
        Predict the next state given current state and sequence context.

        Args:
            current_state: Index of current state
            sequence_features: Recent screenshot features [B, T, D]

        Returns:
            next_state: Predicted next state index
            confidence: Confidence score for the prediction
        """
        with torch.no_grad():
            output = self.forward(sequence_features)
            predictions = output["predictions"]
            transition_probs = output["transition_probs"]

            # Combine model prediction with learned transition probabilities
            combined_scores = (
                predictions + transition_probs[current_state].unsqueeze(0)
            ) / 2

            next_state = torch.argmax(combined_scores, dim=1).item()
            confidence = torch.softmax(combined_scores, dim=1)[0, next_state].item()

        return next_state, confidence

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sequence_predictions: Optional[torch.Tensor] = None,
        sequence_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss for transition prediction.

        Args:
            predictions: Next state predictions [B, num_states]
            targets: Ground truth next states [B]
            sequence_predictions: Per-frame predictions [B, T, num_states]
            sequence_targets: Per-frame ground truth [B, T]

        Returns:
            Dictionary containing:
                - 'total_loss': Combined loss
                - 'next_state_loss': Loss for next state prediction
                - 'sequence_loss': Loss for sequence predictions (if provided)
        """
        # Loss for next state prediction
        next_state_loss = nn.functional.cross_entropy(predictions, targets)

        losses = {"next_state_loss": next_state_loss, "total_loss": next_state_loss}

        # Optional: Add sequence-level loss
        if sequence_predictions is not None and sequence_targets is not None:
            sequence_loss = nn.functional.cross_entropy(
                sequence_predictions.reshape(-1, self.num_states),
                sequence_targets.reshape(-1),
            )
            losses["sequence_loss"] = sequence_loss
            losses["total_loss"] = next_state_loss + 0.5 * sequence_loss

        return losses

    def get_transition_matrix(self) -> torch.Tensor:
        """
        Get the learned transition probability matrix.

        Returns:
            Normalized transition matrix [num_states, num_states]
            where matrix[i, j] represents P(next_state=j | current_state=i)
        """
        return torch.softmax(self.transition_matrix, dim=1)
