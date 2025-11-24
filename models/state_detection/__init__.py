"""
State Detection Module

This module provides models and utilities for ML-based state detection in GUI automation.
State detection analyzes sequences of screenshots to identify distinct application states
and predict transitions between them.

Components:
- RegionProposalNetwork: Proposes candidate state regions from screenshots
- TransitionPredictor: Predicts state transitions based on screenshot sequences
"""

from .region_proposal_network import RegionProposalNetwork
from .transition_predictor import TransitionPredictor

__all__ = [
    'RegionProposalNetwork',
    'TransitionPredictor',
]
