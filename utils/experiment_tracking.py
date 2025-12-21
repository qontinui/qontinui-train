"""
Experiment Tracking Utilities for qontinui-train

This module provides utilities for tracking experiments using Weights & Biases (wandb)
and other experiment tracking platforms.

Author: Joshua Spinak
Project: qontinui-train - Foundation model training for GUI understanding
"""

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

try:
    from pytorch_lightning.loggers import WandbLogger

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False


class ExperimentTracker:
    """
    Unified experiment tracking interface supporting multiple backends.

    Currently supports:
    - Weights & Biases (wandb)
    - TensorBoard (via PyTorch Lightning)
    - MLflow (future)
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        backend: str = "wandb",
        **kwargs,
    ):
        """
        Initialize experiment tracker.

        Args:
            config_path: Path to YAML configuration file
            backend: Tracking backend ("wandb", "tensorboard", "mlflow")
            **kwargs: Override config parameters
        """
        self.backend = backend
        self.config = self._load_config(config_path)
        self.config.update(kwargs)

        self.run = None
        self.logger = None

        if backend == "wandb" and WANDB_AVAILABLE:
            self._init_wandb()

    def _load_config(self, config_path: str | Path | None) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "wandb_config.yaml"

        config_path = Path(config_path)
        if not config_path.exists():
            print(f"Warning: Config file not found at {config_path}. Using defaults.")
            return {}

        with open(config_path) as f:
            config: dict[str, Any] = yaml.safe_load(f)

        result: dict[str, Any] = config.get("wandb", {})
        return result

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if not self.config.get("enabled", True):
            print("wandb logging disabled in config")
            return

        # Extract wandb init parameters
        init_params = {
            "project": self.config.get("project", "qontinui-train"),
            "entity": self.config.get("entity"),
            "name": self.config.get("name"),
            "group": self.config.get("group"),
            "tags": self.config.get("tags", []),
            "notes": self.config.get("notes", ""),
            "mode": self.config.get("mode", "online"),
            "save_code": self.config.get("save_code", True),
            "resume": self.config.get("resume", "allow"),
            "id": self.config.get("resume_id"),
        }

        # Remove None values
        init_params = {k: v for k, v in init_params.items() if v is not None}

        # Initialize wandb
        self.run = wandb.init(**init_params)

        print(f"✓ Initialized wandb run: {self.run.name}")
        print(f"  Project: {self.run.project}")
        print(f"  URL: {self.run.url}")

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None):
        """
        Log metrics to tracking backend.

        Args:
            metrics: Dictionary of metric name -> value
            step: Global step number (optional)
        """
        if self.backend == "wandb" and self.run is not None:
            wandb.log(metrics, step=step)

    def log_hyperparameters(self, params: dict[str, Any]):
        """
        Log hyperparameters.

        Args:
            params: Dictionary of hyperparameter name -> value
        """
        if self.backend == "wandb" and self.run is not None:
            wandb.config.update(params)

    def log_image(
        self,
        key: str,
        image: Any,
        caption: str | None = None,
        step: int | None = None,
    ):
        """
        Log an image.

        Args:
            key: Image key/name
            image: Image (numpy array, PIL Image, or path)
            caption: Optional caption
            step: Global step number
        """
        if self.backend == "wandb" and self.run is not None:
            wandb.log({key: wandb.Image(image, caption=caption)}, step=step)

    def log_predictions(
        self,
        images: list[Any],
        predictions: list[dict[str, Any]],
        ground_truth: list[dict[str, Any]] | None = None,
        step: int | None = None,
    ):
        """
        Log predictions with optional ground truth.

        Args:
            images: List of images
            predictions: List of prediction dictionaries with bbox coordinates
            ground_truth: Optional list of ground truth annotations
            step: Global step number
        """
        if self.backend == "wandb" and self.run is not None:
            # TODO: Implement wandb.Image with bounding boxes
            pass

    def log_model(
        self,
        model_path: str | Path,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Log model checkpoint as artifact.

        Args:
            model_path: Path to model checkpoint
            name: Artifact name
            metadata: Optional metadata dictionary
        """
        if self.backend == "wandb" and self.run is not None:
            artifact = wandb.Artifact(
                name=name or "model", type="model", metadata=metadata or {}
            )
            artifact.add_file(str(model_path))
            self.run.log_artifact(artifact)

    def log_dataset(
        self,
        dataset_path: str | Path,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Log dataset as artifact.

        Args:
            dataset_path: Path to dataset
            name: Artifact name
            metadata: Optional metadata (e.g., num_samples, split)
        """
        if self.backend == "wandb" and self.run is not None:
            artifact = wandb.Artifact(
                name=name or "dataset", type="dataset", metadata=metadata or {}
            )

            # Add directory or file
            dataset_path = Path(dataset_path)
            if dataset_path.is_dir():
                artifact.add_dir(str(dataset_path))
            else:
                artifact.add_file(str(dataset_path))

            self.run.log_artifact(artifact)

    def watch_model(self, model: Any, log: str = "gradients", log_freq: int = 1000):
        """
        Watch model gradients and parameters.

        Args:
            model: PyTorch model
            log: What to log ("gradients", "parameters", "all")
            log_freq: Logging frequency
        """
        if self.backend == "wandb" and self.run is not None:
            wandb.watch(model, log=log, log_freq=log_freq)

    def get_lightning_logger(self) -> Any | None:
        """
        Get PyTorch Lightning logger for this backend.

        Returns:
            Lightning logger instance or None
        """
        if self.backend == "wandb" and LIGHTNING_AVAILABLE and WANDB_AVAILABLE:
            if self.logger is None:
                lightning_config = (
                    self.config.get("lightning", {})
                    .get("logger", {})
                    .get("WandbLogger", {})
                )
                self.logger = WandbLogger(
                    project=lightning_config.get("project", "qontinui-train"),
                    save_dir=lightning_config.get("save_dir", "logs/wandb"),
                    log_model=lightning_config.get("log_model", True),
                    prefix=lightning_config.get("prefix", ""),
                )
            return self.logger

        return None

    def finish(self):
        """Finish tracking session."""
        if self.backend == "wandb" and self.run is not None:
            wandb.finish()
            print("✓ Finished wandb run")


class WandbCallback:
    """
    Callback for logging during training loop.

    Example usage:
        tracker = ExperimentTracker()
        callback = WandbCallback(tracker)

        for epoch in range(num_epochs):
            for batch in dataloader:
                loss = train_step(batch)
                callback.on_batch_end({"loss": loss}, step)
    """

    def __init__(self, tracker: ExperimentTracker):
        """
        Initialize callback.

        Args:
            tracker: ExperimentTracker instance
        """
        self.tracker = tracker

    def on_train_begin(self, config: dict[str, Any]):
        """Called at the beginning of training."""
        self.tracker.log_hyperparameters(config)

    def on_epoch_begin(self, epoch: int):
        """Called at the beginning of each epoch."""
        pass

    def on_batch_end(self, metrics: dict[str, Any], step: int):
        """Called at the end of each batch."""
        self.tracker.log_metrics(metrics, step=step)

    def on_epoch_end(self, metrics: dict[str, Any], epoch: int):
        """Called at the end of each epoch."""
        self.tracker.log_metrics(metrics, step=epoch)

    def on_validation_end(self, metrics: dict[str, Any], epoch: int):
        """Called at the end of validation."""
        self.tracker.log_metrics(metrics, step=epoch)

    def on_train_end(self):
        """Called at the end of training."""
        self.tracker.finish()


def setup_experiment_tracking(
    config_path: str | Path | None = None, backend: str = "wandb", **kwargs
) -> ExperimentTracker:
    """
    Convenience function to set up experiment tracking.

    Args:
        config_path: Path to configuration file
        backend: Tracking backend
        **kwargs: Override config parameters

    Returns:
        ExperimentTracker instance

    Example:
        tracker = setup_experiment_tracking(
            project="qontinui-train",
            name="vit_base_mae_pretrain",
            tags=["mae", "baseline", "vit-base"]
        )

        # Log hyperparameters
        tracker.log_hyperparameters(config)

        # Training loop
        for epoch in range(epochs):
            metrics = train_epoch()
            tracker.log_metrics(metrics, step=epoch)

        # Finish
        tracker.finish()
    """
    return ExperimentTracker(config_path=config_path, backend=backend, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("Experiment Tracking Setup")
    print("=" * 50)

    # Initialize tracker
    tracker = setup_experiment_tracking(
        project="qontinui-train",
        name="test_run",
        tags=["test"],
        mode="disabled",  # Don't actually log for this test
    )

    # Log some metrics
    tracker.log_metrics({"loss": 0.5, "accuracy": 0.85}, step=1)
    tracker.log_hyperparameters({"learning_rate": 1e-4, "batch_size": 32})

    print("\n✓ Experiment tracking setup complete")
    print("\nTo use wandb:")
    print("1. Install: pip install wandb")
    print("2. Login: wandb login")
    print("3. Update config: configs/wandb_config.yaml")
    print("4. Run training with tracker enabled")
