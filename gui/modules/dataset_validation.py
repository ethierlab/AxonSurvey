import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
from config import Config
import sys

# Need to import TracingChecker
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.dataprep.TracingChecker import TracingChecker
from modules.data_navigation import get_rat_regions, get_rat_subregions

@dataclass
class ImageInfo:
    dataset_name: str
    folder_name: str
    rat: str
    bregma: str
    region: str
    original_path: str
    has_tracing: bool
    has_mask: bool
    has_image: bool

def parse_info_txt(info_path: Path) -> dict:
    if not info_path.exists():
        return {}
    with open(info_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Based on TracingChecker and example:
    # Line 1: original file path
    # Line 3: rat
    # Line 4: bregma (slice)
    # Line 5: region
    
    return {
        'original_path': lines[1] if len(lines) > 1 else "",
        'rat': lines[3] if len(lines) > 3 else "",
        'bregma': lines[4] if len(lines) > 4 else "",
        'region': lines[5] if len(lines) > 5 else ""
    }

def get_balance_category(counts_dict):
    if not counts_dict:
        return "empty"
    values = list(counts_dict.values())
    min_val = min(values)
    max_val = max(values)
    
    if min_val == max_val:
        return "equal"
        
    # roughly equal: approximately ±20%
    mean_val = sum(values) / len(values)
    if mean_val > 0 and all(abs(v - mean_val) / mean_val <= 0.20 for v in values):
        return "roughly equal"
        
    return "unbalanced"

def get_rat_balance_category(rat_counts):
    if not rat_counts:
        return "empty"
    
    proportions = []
    for rat, count in rat_counts.items():
        try:
            slices = get_rat_regions(rat)
            underlying_regions = 0
            for s in slices:
                underlying_regions += len(get_rat_subregions(rat, s))
                
            if underlying_regions == 0:
                proportions.append(0)
            else:
                proportions.append(count / underlying_regions)
        except:
            proportions.append(count)
            
    if not proportions:
        return "empty"
        
    min_prop = min(proportions)
    max_prop = max(proportions)
    
    if min_prop == max_prop:
        return "100% proportional"
        
    mean_prop = sum(proportions) / len(proportions)
    if mean_prop > 0 and all(abs(p - mean_prop) / mean_prop <= 0.20 for p in proportions):
        return "roughly proportional"
        
    return "unbalanced"

def get_dataset_statistics():
    datasets = {
        'Training': Config.TRAINING_DIR,
        'Testing': Config.TESTING_DIR
    }
    
    problems_summary = {}
    dataset_statistics = {}
    
    # Project scans verification
    from modules.data_navigation import scan_available_rats
    all_rats = scan_available_rats()
    project_scans_verification = []
    rat_underlying_counts = {}
    
    for rat in all_rats:
        slices = get_rat_regions(rat)
        total_regions = 0
        for s in slices:
            total_regions += len(get_rat_subregions(rat, s))
        rat_underlying_counts[rat] = total_regions
        
    if rat_underlying_counts:
        max_regions = max(rat_underlying_counts.values())
        for rat, count in rat_underlying_counts.items():
            if count < max_regions:
                project_scans_verification.append({
                    'rat': rat,
                    'count': count,
                    'max': max_regions,
                    'message': f"Rat {rat} has fewer regions ({count}) than the maximum found ({max_regions}). This may be due to a different number of bregma slices."
                })
    
    for ds_name, ds_path in datasets.items():
        if not ds_path.exists():
            continue
            
        checker = TracingChecker(str(ds_path))
        files_in_root, missing_images, missing_tracings, missing_channels, missing_masks, missing_info, extra_stuff = checker.get_problems()
        
        problems_summary[ds_name] = {
            'missing_images': missing_images,
            'missing_tracings': missing_tracings,
            'missing_masks': missing_masks,
            'missing_info': missing_info,
            'extra_stuff': extra_stuff,
            'files_in_root': files_in_root
        }
        
        folders = checker.get_folders_in_root()
        ds_images = []
        for fldr in folders:
            folder_path = ds_path / fldr
            info_path = folder_path / checker.info_file_name
            
            info_data = parse_info_txt(info_path)
            
            img_info = ImageInfo(
                dataset_name=ds_name,
                folder_name=fldr,
                rat=info_data.get('rat', 'Unknown'),
                bregma=info_data.get('bregma', 'Unknown'),
                region=info_data.get('region', 'Unknown'),
                original_path=info_data.get('original_path', ''),
                has_tracing=fldr not in missing_tracings,
                has_mask=fldr not in missing_masks,
                has_image=fldr not in missing_images
            )
            ds_images.append(img_info)
            
        # Compute statistics
        rat_counts = {}
        region_counts = {}
        slice_counts = {}
        rat_bregma_counts = {}
        
        for img in ds_images:
            if img.rat:
                rat_counts[img.rat] = rat_counts.get(img.rat, 0) + 1
            if img.region:
                region_counts[img.region] = region_counts.get(img.region, 0) + 1
            if img.bregma:
                slice_counts[img.bregma] = slice_counts.get(img.bregma, 0) + 1
                
            if img.rat and img.bregma:
                key = f"{img.rat} - {img.bregma}"
                rat_bregma_counts[key] = rat_bregma_counts.get(key, 0) + 1
            
        stats = {
            'rat_distribution': {
                'counts': rat_counts,
                'status': get_rat_balance_category(rat_counts)
            },
            'region_distribution': {
                'counts': region_counts,
                'status': get_balance_category(region_counts)
            },
            'slice_distribution': {
                'counts': slice_counts,
                'status': get_balance_category(slice_counts)
            },
            'rat_bregma_distribution': {
                'counts': rat_bregma_counts,
                'status': get_balance_category(rat_bregma_counts)
            }
        }
        
        dataset_statistics[ds_name] = {
            'statistics': stats,
            'total_images': len(ds_images)
        }
    
    return {
        'problems': problems_summary,
        'datasets': dataset_statistics,
        'project_scans_verification': project_scans_verification
    }
