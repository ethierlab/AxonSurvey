#!/usr/bin/env python3
"""
Neural Network Training Script

This script trains a UNet model for axon segmentation on your labeled dataset.

Usage:
    python scripts/3-train_model.py --train-dir ./data/tracings/train --test-dir ./data/tracings/test --output ./data/trained_models/default_model.pth --epochs 50
    
    python scripts/3-train_model.py --train-dir ./data/tracings/train --test-dir ./data/tracings/test --output ./data/trained_models/default_model.pth --epochs 50 --batch-size 32 --learning-rate 0.0001

For more information, see the README.md in the scripts folder.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.NNs.Dataset import SimpleAxonDataset
from src.NNs.Unet import UNetModel
from src.NNs.training import train_and_save_Unet, save_model_metadata
from src.dataprep.TracingChecker import TracingChecker
from transformers import get_cosine_schedule_with_warmup


def check_dataset(dataset_path, dataset_name):
    """Check if dataset is valid."""
    print(f"\nChecking {dataset_name} dataset...")
    checker = TracingChecker(dataset_path)
    checker.check()
    
    if not checker.is_valid():
        raise ValueError(f"{dataset_name} dataset is not valid. Please check the dataset structure.")
    
    labeled_ratio = checker.get_labelled_ratio()
    print(f"  Labeled ratio: {labeled_ratio:.2%}")
    
    if labeled_ratio == 0.0:
        raise ValueError(f"{dataset_name} dataset has no labeled data. Please add tracings.")
    
    if labeled_ratio < 0.5:
        print(f"  WARNING: Only {labeled_ratio:.2%} of samples are labeled. Consider labeling more samples.")
    
    return checker


def train_unet_model(train_dir, test_dir, output_path, epochs, batch_size, learning_rate, 
                     input_size, display_epochs, use_scheduler=True):
    """
    Train a UNet model for axon segmentation.
    
    Args:
        train_dir: Directory containing training dataset
        test_dir: Directory containing test dataset
        output_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        input_size: Input image size (assumes square)
        display_epochs: Display progress every N epochs
        use_scheduler: Whether to use cosine learning rate scheduler
    """
    print("=" * 60)
    print("Neural Network Training")
    print("=" * 60)
    
    # Check datasets
    train_checker = check_dataset(train_dir, "Training")
    test_checker = check_dataset(test_dir, "Test")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cpu":
        print("  WARNING: CUDA not available. Training will be slower on CPU.")
    
    # Create datasets
    print(f"\nLoading datasets...")
    print(f"  Input size: {input_size}x{input_size}")
    
    train_dataset = SimpleAxonDataset(train_dir, input_size=input_size)
    test_dataset = SimpleAxonDataset(test_dir, input_size=input_size)
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create model
    print(f"\nCreating UNet model...")
    model_type = UNetModel
    
    # Setup loss function (using soft_dice_cldice if available, otherwise BCE)
    try:
        from clDice.cldice_loss.pytorch.cldice import soft_dice_cldice
        criterion = soft_dice_cldice(iter_=10, alpha=0.3, smooth=0.0)
        print("  Using soft_dice_cldice loss function")
    except ImportError:
        from torch.nn import BCEWithLogitsLoss
        criterion = BCEWithLogitsLoss()
        print("  Using BCEWithLogitsLoss (clDice not available)")
    
    # Setup scheduler
    scheduler_func = None
    if use_scheduler:
        scheduler_func = get_cosine_schedule_with_warmup
        print("  Using cosine learning rate scheduler with warmup")
    
    # Training parameters
    print(f"\nTraining parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Display every: {display_epochs} epochs")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nCreated output directory: {output_dir}")
    
    # Train model
    print(f"\nStarting training...")
    print("=" * 60)
    
    try:
        train_and_save_Unet(
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            n_epochs=epochs,
            model_type=model_type,
            model_path=output_path,
            criterion=criterion,
            lr=learning_rate,
            batch_size=batch_size,
            n_epochs_display=display_epochs,
            display_loss=True,
            make_schedular_function=scheduler_func
        )
        
        # Save metadata
        metadata = {
            "model_path": str(output_path),
            "train_dir": str(train_dir),
            "test_dir": str(test_dir),
            "n_epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "input_size": input_size,
            "criterion": str(criterion),
            "scheduler": "cosine_with_warmup" if use_scheduler else "none",
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Determine metadata directory relative to output path
        metadata_dir = os.path.join(os.path.dirname(os.path.dirname(output_path)), "training_model_metadata")
        save_model_metadata(output_path, metadata, metadata_dir=metadata_dir)
        
        print("=" * 60)
        print(f"✓ Training completed successfully!")
        print(f"  Model saved to: {output_path}")
        print(f"\nNext steps:")
        print(f"  1. Use this model for inference: python scripts/4-run_inference.py --model {output_path}")
        print(f"  2. Or use it for NN-based sampling: python scripts/2-sample_data.py --nn --model-path {output_path}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Train a UNet model for axon segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default parameters
  python scripts/3-train_model.py --train-dir ./data/tracings/train --test-dir ./data/tracings/test --output ./data/trained_models/default_model.pth --epochs 50
  
  # Training with custom parameters
  python scripts/3-train_model.py --train-dir ./data/tracings/train --test-dir ./data/tracings/test --output ./data/trained_models/default_model.pth --epochs 100 --batch-size 32 --learning-rate 0.0001 --input-size 256
  
  # Quick test training (few epochs)
  python scripts/3-train_model.py --train-dir ./data/tracings/train --test-dir ./data/tracings/test --output ./data/trained_models/test_model.pth --epochs 5

Note: The training and test directories should contain labeled datasets created by scripts/2-sample_data.py
        """
    )
    
    parser.add_argument(
        '--train-dir',
        type=str,
        required=True,
        help='Directory containing training dataset'
    )
    
    parser.add_argument(
        '--test-dir',
        type=str,
        required=True,
        help='Directory containing test dataset'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./data/trained_models/default_model.pth',
        help='Output path for trained model (.pth file) (default: ./data/trained_models/default_model.pth)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.0001,
        help='Learning rate (default: 0.0001)'
    )
    
    parser.add_argument(
        '--input-size',
        type=int,
        default=128,
        help='Input image size, assumes square (default: 128)'
    )
    
    parser.add_argument(
        '--display-epochs',
        type=int,
        default=5,
        help='Display progress every N epochs (default: 5)'
    )
    
    parser.add_argument(
        '--no-scheduler',
        action='store_true',
        help='Disable learning rate scheduler'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.train_dir):
        print(f"Error: Training directory does not exist: {args.train_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory does not exist: {args.test_dir}")
        sys.exit(1)
    
    if args.epochs <= 0:
        parser.error("--epochs must be a positive integer")
    
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")
    
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be a positive number")
    
    if args.input_size <= 0:
        parser.error("--input-size must be a positive integer")
    
    # Train model
    train_unet_model(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        input_size=args.input_size,
        display_epochs=args.display_epochs,
        use_scheduler=not args.no_scheduler
    )


if __name__ == '__main__':
    main()

