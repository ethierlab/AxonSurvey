#!/usr/bin/env python3
"""
Inference Script

This script applies a trained model to all images in your Images Dataset for segmentation.
The results can then be analyzed using the Axon Survey GUI.

Input Requirements:
- Trained model file (.pth) (input)
- Valid Images Dataset directory containing raw full-size scans (input)

Usage:
    python scripts/4-run_inference.py --model ./data/trained_models/default_model.pth --input ./data/project_scans --output ./data/segmented_images
    
    python scripts/4-run_inference.py --model ./data/trained_models/default_model.pth --input ./data/project_scans --output ./data/segmented_images --input-size 128 --batch-size 16
    
    # Minimal test
    python scripts/4-run_inference.py --model ./data/trained_models/test_model.pth --input ./data/test_scans --output ./data/test_segmented --input-size 32

For more information, see the README.md in the scripts folder.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.tracers.DLTracer import DLTracer
from src.NNs.Unet import UNetModel
from src.utils.imageio import tif_to_numpy, numpy_to_tif, generate_image_outer_mask
from src.dataprep.DataReader import DataReader


def process_single_image(image_path, tracer, output_dir, channel="th"):
    """
    Process a single image with the tracer and save segmentation.
    
    Args:
        image_path: Path to input image
        tracer: DLTracer instance
        output_dir: Directory to save segmented image
        channel: Channel name to use
    """
    try:
        # Load image
        full_image_path = os.path.join(image_path, f"{channel}.tif")
        if not os.path.exists(full_image_path):
            print(f"  Skipping {image_path}: {channel}.tif not found")
            return False
        
        # Load image (tif_to_numpy returns float32, 3D by default)
        image = tif_to_numpy(full_image_path, output_dims=3)
        
        # Ensure image is 3D (H, W, C) and take first channel if multi-channel
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        elif image.shape[2] > 1:
            image = image[:, :, :1]  # Take first channel if multi-channel
        
        # Ensure float32 dtype
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Generate mask
        mask = generate_image_outer_mask(image)
        
        # Trace
        trace = tracer.trace(image, mask).squeeze()
        
        # Save segmentation
        output_path = os.path.join(output_dir, "segmentation.tif")
        os.makedirs(output_dir, exist_ok=True)
        numpy_to_tif(trace, output_path)
        
        return True
    except Exception as e:
        print(f"  Error processing {image_path}: {e}")
        return False


def run_inference(model_path, input_dir, output_dir, input_size=128, channel="th"):
    """
    Run inference on all images in the input directory.
    
    Args:
        model_path: Path to trained model (.pth file)
        input_dir: Directory containing the Images Dataset
        output_dir: Directory to save segmented images
        input_size: Input size for the model
        channel: Channel name to use
    """
    print("=" * 60)
    print("Running Inference")
    print("=" * 60)
    
    # Validate inputs
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")
    
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory not found: {input_dir}")
    
    # Create tracer
    print(f"\nLoading model: {model_path}")
    print(f"  Input size: {input_size}x{input_size}")
    
    tracer = DLTracer(
        model_path=model_path,
        model_type=UNetModel,
        img_input_size=input_size,
        tracer_name="inference_tracer"
    )
    
    # Get all image paths
    print(f"\nScanning input directory: {input_dir}")
    dr = DataReader(input_dir)
    
    if not dr.read_dir_is_valid():
        raise ValueError(f"Input directory structure is not valid: {input_dir}")
    
    image_paths = dr.get_paths()
    print(f"  Found {len(image_paths)} image folders")
    
    # Process each image
    print(f"\nProcessing images...")
    print(f"  Output directory: {output_dir}")
    
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"  [{i}/{len(image_paths)}] Processing {image_path}...", end=" ")
        
        # Create corresponding output path
        rel_path = os.path.relpath(image_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        if process_single_image(image_path, tracer, output_path, channel):
            print("✓")
            successful += 1
        else:
            print("✗")
            failed += 1
    
    # Summary
    print("=" * 60)
    print(f"✓ Inference completed!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Start the Axon Survey GUI: python gui/app.py")
    print(f"  2. Navigate to the compare page to analyze segmented images")
    print(f"  3. Download comparison data for detailed analysis")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on all images in your Images Dataset using a trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python scripts/4-run_inference.py --model ./data/trained_models/default_model.pth --input ./data/project_scans --output ./data/segmented_images
  
  # Inference with custom parameters
  python scripts/4-run_inference.py --model ./data/trained_models/default_model.pth --input ./data/project_scans --output ./data/segmented_images --input-size 256 --channel th
  
  # Process specific channel
  python scripts/4-run_inference.py --model ./data/trained_models/default_model.pth --input ./data/project_scans --output ./data/segmented_images --channel dbh
  
  # Minimal test (fast execution, minimal resources)
  python scripts/4-run_inference.py --model ./data/trained_models/test_model.pth --input ./data/test_scans --output ./data/test_segmented --input-size 32

Note: The input directory should follow the structure created by scripts/1-setup_folder_structure.py
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='./data/trained_models/default_model.pth',
        help='Path to trained model (.pth file) (default: ./data/trained_models/default_model.pth)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing the Images Dataset'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for segmented images'
    )
    
    parser.add_argument(
        '--input-size',
        type=int,
        default=128,
        help='Input size for the model (default: 128)'
    )
    
    parser.add_argument(
        '--channel',
        type=str,
        default='th',
        help='Channel name to use (default: th)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input_size <= 0:
        parser.error("--input-size must be a positive integer")
    
    # Run inference
    try:
        run_inference(
            model_path=args.model,
            input_dir=args.input,
            output_dir=args.output,
            input_size=args.input_size,
            channel=args.channel
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

