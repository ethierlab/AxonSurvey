#!/usr/bin/env python3
"""
Setup Folder Structure Script

This script creates the directory structure needed for your Images Dataset (raw full-size rat brain scans).
It follows the structure: project_scans/rat_id/bregma/region/

Input Requirements:
- Configuration JSON file (located at scripts/configs/image_dataset_config.json)

Usage:
    python scripts/1-setup_folder_structure.py
    
    # Minimal test
    # Edit scripts/configs/image_dataset_config.json to have minimal rats/bregmas/regions, then run:
    # python scripts/1-setup_folder_structure.py

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
    print("Using configuration from scripts/configs/image_dataset_config.json")
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'image_dataset_config.json')
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
        
    config = load_config_from_json(config_path)
    rat_list = config['rat_list']
    bregma_dict = config['bregma_dict']
    subregion_list = config['subregion_list']
    base_path = config.get('base_path', './data/project_scans')
    
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

