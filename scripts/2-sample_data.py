#!/usr/bin/env python3
"""
Data Sampling Script

This script samples image patches from your project data to create training and test datasets.
It supports random sampling and (future) neural network-based sampling.

Usage:
    # Random sampling
    python scripts/2-sample_data.py --random --input ./data/project_scans --output ./data/tracings/train --size 100 --patch-size 128
    
    # Neural network-based sampling (future feature)
    python scripts/2-sample_data.py --nn --model-path ./data/trained_models/default_model.pth --input ./data/project_scans --output ./data/tracings/train --size 100

For more information, see the README.md in the scripts folder.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataprep.SampleSaver import SampleSaver
from src.dataprep.SamplingStrategies import SRS
from src.experiments.RatGroup import RatGroup, ALL_RATS, ALL_REGIONS


def create_random_sampling_dataset(input_dir, output_dir, sample_size, patch_size, channel="th", 
                                   stratify_regions=True, groups=None):
    """
    Create a dataset using random sampling.
    
    Args:
        input_dir: Directory containing raw project images
        output_dir: Directory to save sampled dataset
        sample_size: Number of samples to create
        patch_size: Size of each patch (assumes square patches)
        channel: Channel name to sample from (default: "th")
        stratify_regions: Whether to stratify by region (default: True)
        groups: RatGroup or list of RatGroups to sample from (None = all)
    """
    print(f"Creating random sampling dataset...")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Sample size: {sample_size}")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Channel: {channel}")
    
    # Ensure output directory exists and is empty
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise ValueError(f"Output directory {output_dir} is not empty. Please use a new directory or remove existing files.")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sampling strategy
    sample_dimensions = (patch_size, patch_size)
    
    if groups is None:
        groups = RatGroup(rats=ALL_RATS, regions=ALL_REGIONS)
    
    sampling_strategy = SRS(
        project_raw_image_dir=input_dir,
        groups=groups,
        channel=channel,
        sample_dimensions=sample_dimensions,
        stratify_regions=stratify_regions,
        stratify_group=True
    )
    
    # Create sample saver and generate dataset
    sample_saver = SampleSaver(
        new_dataset_dir=output_dir,
        sampling_strategy=sampling_strategy
    )
    
    print(f"\nSampling {sample_size} patches...")
    sample_saver.create_dataset(sample_size)
    print(f"✓ Dataset created successfully in {output_dir}")
    
    print("\nNext steps:")
    print("  1. Manually trace axons in the sampled images using NeuronJ or similar tool")
    print("  2. Save tracings as 'tracings.tif' in each sample folder")
    print("  3. Run scripts/3-train_model.py to train a neural network on your traced data")


def create_nn_sampling_dataset(input_dir, output_dir, model_path, sample_size, patch_size, 
                               channel="th", groups=None):
    """
    Create a dataset using neural network-based sampling (informative sampling).
    
    NOTE: This is a placeholder for future implementation. Currently falls back to random sampling.
    
    Args:
        input_dir: Directory containing raw project images
        output_dir: Directory to save sampled dataset
        model_path: Path to trained model (.pth file)
        sample_size: Number of samples to create
        patch_size: Size of each patch
        channel: Channel name to sample from
        groups: RatGroup or list of RatGroups to sample from
    """
    print("WARNING: Neural network-based sampling is not yet fully implemented.")
    print("Falling back to random sampling for now.")
    print(f"Model path provided: {model_path}")
    
    # For now, use random sampling
    create_random_sampling_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        sample_size=sample_size,
        patch_size=patch_size,
        channel=channel,
        groups=groups
    )
    
    print("\nFuture implementation will:")
    print("  - Use the model to identify informative regions")
    print("  - Sample patches where the model is uncertain or where features are interesting")
    print("  - Provide a GUI with sliders for adjusting sampling parameters")


def main():
    parser = argparse.ArgumentParser(
        description="Sample image patches to create training/test datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random sampling for training data
  python scripts/2-sample_data.py --random --input ./data/project_scans --output ./data/tracings/train --size 200 --patch-size 128
  
  # Random sampling for test data
  python scripts/2-sample_data.py --random --input ./data/project_scans --output ./data/tracings/test --size 100 --patch-size 128
  
  # Neural network-based sampling (when implemented)
  python scripts/2-sample_data.py --nn --model-path ./data/trained_models/default_model.pth --input ./data/project_scans --output ./data/tracings/train --size 200

Note: The output directory must be empty or not exist.
        """
    )
    
    # Sampling method selection
    sampling_group = parser.add_mutually_exclusive_group(required=True)
    sampling_group.add_argument(
        '--random',
        action='store_true',
        help='Use random sampling strategy'
    )
    sampling_group.add_argument(
        '--nn',
        action='store_true',
        help='Use neural network-based sampling (requires --model-path)'
    )
    
    # Common arguments
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing raw project images'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for sampled dataset'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        required=True,
        help='Number of samples to create'
    )
    
    parser.add_argument(
        '--patch-size',
        type=int,
        default=128,
        help='Size of each patch (assumes square, default: 128)'
    )
    
    parser.add_argument(
        '--channel',
        type=str,
        default='th',
        help='Channel name to sample from (default: th)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='./data/trained_models/default_model.pth',
        help='Path to trained model (.pth file) for NN-based sampling (default: ./data/trained_models/default_model.pth)'
    )
    
    parser.add_argument(
        '--no-stratify',
        action='store_true',
        help='Disable region stratification (sample proportionally to area)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input):
        print(f"Error: Input directory does not exist: {args.input}")
        sys.exit(1)
    
    if args.nn and not args.model_path:
        parser.error("--nn requires --model-path to be specified")
    
    if args.size <= 0:
        parser.error("--size must be a positive integer")
    
    if args.patch_size <= 0:
        parser.error("--patch-size must be a positive integer")
    
    # Create dataset
    try:
        if args.random:
            create_random_sampling_dataset(
                input_dir=args.input,
                output_dir=args.output,
                sample_size=args.size,
                patch_size=args.patch_size,
                channel=args.channel,
                stratify_regions=not args.no_stratify
            )
        elif args.nn:
            create_nn_sampling_dataset(
                input_dir=args.input,
                output_dir=args.output,
                model_path=args.model_path,
                sample_size=args.size,
                patch_size=args.patch_size,
                channel=args.channel
            )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

