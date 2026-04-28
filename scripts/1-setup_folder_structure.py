#!/usr/bin/env python3
"""
Setup Folder Structure Script

This script creates the directory structure needed for your Images Dataset (raw full-size rat brain scans).
It follows the structure: project_scans/rat_id/bregma/region/

Input Requirements:
- Configuration JSON file (if using --config)

Usage:
    python scripts/1-setup_folder_structure.py --config config.json
    python scripts/1-setup_folder_structure.py --rats rat301,rat302 --bregmas b516,b468 --regions contra_inner,contra_outer --output ./data/project_scans
    
    # Minimal test
    python scripts/1-setup_folder_structure.py --rats test_rat --bregmas b0 --regions test_region --output ./data/test_scans

For more information, see the README.md in the scripts folder.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataprep.FileOperations import SpecFileStructuror


def load_config_from_json(config_path):
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_folder_structure(rat_list, bregma_dict, subregion_list, base_path):
    """
    Create folder structure for the Images Dataset.
    
    Args:
        rat_list: List of rat IDs (e.g., ["rat301", "rat302"])
        bregma_dict: Dictionary mapping rat IDs to lists of bregma values
                    (e.g., {"rat301": ["b516", "b468"], "rat302": ["b252"]})
        subregion_list: List of region names (e.g., ["contra_inner", "contra_outer"])
        base_path: Base directory path for the Images Dataset
    """
    # Ensure base path exists
    os.makedirs(base_path, exist_ok=True)
    
    # Create file structurer and make folders
    file_structurer = SpecFileStructuror(
        rat_list=rat_list,
        bregma_dict=bregma_dict,
        subregion_list=subregion_list,
        base_path=base_path
    )
    
    print(f"Creating folder structure in: {base_path}")
    file_structurer.make_folders()
    print("✓ Folder structure created successfully!")
    
    # Print summary
    print("\nCreated structure:")
    for rat in rat_list:
        for bregma in bregma_dict.get(rat, []):
            for subregion in subregion_list:
                path = os.path.join(base_path, rat, bregma, subregion)
                print(f"  {path}")


def create_example_config(output_path="example_config.json"):
    """Create an example configuration file."""
    example_config = {
        "rat_list": ["rat301", "rat302", "rat303"],
        "bregma_dict": {
            "rat301": ["b516", "b468", "b420"],
            "rat302": ["b252", "b521"],
            "rat303": ["b252", "b521", "b533"]
        },
        "subregion_list": ["contra_inner", "contra_outer", "ipsi_inner", "ipsi_outer"],
        "base_path": "./data/project_scans"
    }
    
    with open(output_path, 'w') as f:
        json.dump(example_config, f, indent=4)
    
    print(f"Example configuration file created: {output_path}")
    print("\nYou can edit this file and use it with --config option.")


def main():
    parser = argparse.ArgumentParser(
        description="Create folder structure for your Images Dataset (raw full-size scans)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using a JSON config file
  python scripts/1-setup_folder_structure.py --config my_config.json
  
  # Using command-line arguments
  python scripts/1-setup_folder_structure.py --rats rat301,rat302 --bregmas b516,b468 --regions contra_inner,contra_outer --output ./data/project_scans
  
  # Create example config file
  python scripts/1-setup_folder_structure.py --create-example-config
  
  # Minimal test (fast execution, minimal resources)
  python scripts/1-setup_folder_structure.py --rats test_rat --bregmas b0 --regions test_region --output ./data/test_scans

Note: The bregma_dict format in JSON should be:
{
  "rat301": ["b516", "b468"],
  "rat302": ["b252"]
}
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file'
    )
    
    parser.add_argument(
        '--rats',
        type=str,
        help='Comma-separated list of rat IDs (e.g., "rat301,rat302")'
    )
    
    parser.add_argument(
        '--bregmas',
        type=str,
        help='Comma-separated list of bregma values (applied to all rats)'
    )
    
    parser.add_argument(
        '--regions',
        type=str,
        help='Comma-separated list of region names (e.g., "contra_inner,contra_outer")'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./data/project_scans',
        help='Base output directory path (default: ./data/project_scans)'
    )
    
    parser.add_argument(
        '--create-example-config',
        action='store_true',
        help='Create an example configuration file and exit'
    )
    
    args = parser.parse_args()
    
    # Create example config if requested
    if args.create_example_config:
        create_example_config()
        return
    
    # Load configuration
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Configuration file not found: {args.config}")
            sys.exit(1)
        config = load_config_from_json(args.config)
        rat_list = config['rat_list']
        bregma_dict = config['bregma_dict']
        subregion_list = config['subregion_list']
        base_path = config.get('base_path', args.output)
    else:
        # Parse command-line arguments
        if not args.rats or not args.bregmas or not args.regions:
            parser.error("Either --config or all of --rats, --bregmas, and --regions must be provided")
        
        rat_list = [r.strip() for r in args.rats.split(',')]
        bregma_list = [b.strip() for b in args.bregmas.split(',')]
        subregion_list = [r.strip() for r in args.regions.split(',')]
        
        # Create bregma_dict with same bregmas for all rats
        bregma_dict = {rat: bregma_list for rat in rat_list}
        base_path = args.output
    
    # Create folder structure
    try:
        create_folder_structure(rat_list, bregma_dict, subregion_list, base_path)
    print("\n✓ Setup complete! You can now add your raw full-size scans to the created folders.")
    print("\nNext steps:")
    print("  1. Add your .tif images to the appropriate folders in your Images Dataset")
    print("  2. Run the sampling script to create training/test Tracings Datasets")
        print("  3. See README.md for detailed instructions")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

