"""
🐭 Rat Brain GUI - Flask Application

Web app frontend to explore structured rat brain image datasets and interact 
with existing segmentation + feature map models. Built for non-technical local users.

"""
import matplotlib
matplotlib.use('Agg')

import traceback

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import logging
import json

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.imageio import numpy_to_tif, tif_to_numpy
from src.tracers.DLTracer import DLTracer
from src.NNs.Unet import UNetModel
from src.utils.imageio import generate_image_outer_mask
from src.experiments.RatGroup import RatGroup, ALL_RATS, ALL_REGIONS

from config import Config
from modules.run_quantification import run_quantification
from modules.data_navigation import scan_available_rats, get_rat_regions, \
                            get_rat_subregions, get_rat_metadata, get_directories_with_tif_images, \
                            scan_available_experiments
from modules.data_loading import convert_tif_to_jpg_and_save, validate_safe_path, \
                        get_region_cached_path, get_cached_segmentation_path, \
                        get_cached_feature_map_path, get_comparison_data
from modules.dataset_validation import get_dataset_statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Flask routes
@app.route('/')
def index():
    """Index page - shows list of available rats and experiments"""
    error = None
    if not Config.DATA_DIR.exists():
        error = f"Data directory not found: {Config.DATA_DIR}. Please check your config.json."
    elif not Config.CACHE_DIR.exists():
        error = f"Cache directory not found: {Config.CACHE_DIR}. Please check your config.json."
    elif not Config.USED_SEGMENTATION_MODEL_PATH.exists():
        error = f"Segmentation model not found at {Config.USED_SEGMENTATION_MODEL_PATH}. You should train a model before doing this step. Please check your config.json."
        
    rats = scan_available_rats()
    experiments = scan_available_experiments()
    return render_template('index.html', rats=rats, experiments=experiments, error=error)


@app.route('/quicktrace', methods=['GET', 'POST'])
def quicktrace_page():
    """Quicktrace page - allow user to submit image path to trace"""
    if request.method == 'POST':
        image_path = request.form.get('image_path')
        if image_path:
            image_path = image_path.strip('\"\'') # Remove surrounding quotes
        if not image_path:
            return render_template('quicktrace.html', error="No file path provided.")

        try:
            # Validate the path is safe
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                return render_template('quicktrace.html', error="File does not exist.")
            image_path = str(image_path_obj)

            original_image_jpg_path = os.path.join(app.static_folder, 'quicktrace', 'original_image.jpg')
            convert_tif_to_jpg_and_save(Path(image_path), original_image_jpg_path, use_raw_path=True)
            
            # Generate tracing
            logger.info(f"Starting image tracing for {image_path}")
            import time
            start_time = time.time()

            image = tif_to_numpy(image_path)[:, :, :1]
            logger.info(f"tif_to_numpy completed in {time.time() - start_time:.2f} seconds.")
            start_time = time.time()

            mask = generate_image_outer_mask(image)
            logger.info(f"generate_image_outer_mask completed in {time.time() - start_time:.2f} seconds.")
            start_time = time.time()

            if not Config.USED_SEGMENTATION_MODEL_PATH.exists():
                return render_template('quicktrace.html', error=f"Segmentation model not found at {Config.USED_SEGMENTATION_MODEL_PATH}. You should train a model before doing this step. Please check your config.json.")
            
            tracer = DLTracer(Config.USED_SEGMENTATION_MODEL_PATH, UNetModel, 128, tracer_name="gui_tracer")
            trace = tracer.trace(image, mask).squeeze()
            logger.info(f"DLTracer.trace completed in {time.time() - start_time:.2f} seconds.")
            start_time = time.time()

            output_tif_path = os.path.join(app.static_folder, 'quicktrace', 'traced_result.tif')
            numpy_to_tif(trace, output_tif_path)
            logger.info(f"numpy_to_tif completed in {time.time() - start_time:.2f} seconds.")
            start_time = time.time()

            output_trace_jpg_path = os.path.join(app.static_folder, 'quicktrace', 'traced_result.jpg')
            convert_tif_to_jpg_and_save(Path(output_tif_path), output_trace_jpg_path, use_raw_path=True)
            logger.info(f"convert_tif_to_jpg_and_save for traced result completed in {time.time() - start_time:.2f} seconds.")

            return render_template('quicktrace.html', 
                                original_image=os.path.abspath(original_image_jpg_path),
                                result_image=os.path.abspath(output_trace_jpg_path),
                                original_image_file_path=os.path.abspath(image_path), 
                                high_quality_file_path=os.path.abspath(output_tif_path))


        except MemoryError as e:
            logger.error(f"MemoryError during quicktrace: {e}. This usually means the image is too large for the server's available memory.", exc_info=True)(f"MemoryError during quicktrace: {e}. This usually means the image is too large for the server's available memory.", exc_info=True)
            return render_template('quicktrace.html', error="The image is too large to process. Please try a smaller image or reduce its resolution.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during quicktrace: {e}", exc_info=True)
            return render_template('quicktrace.html', error=f"An unexpected error occurred during tracing: {e}. Please check server logs for details.")

    return render_template('quicktrace.html')

@app.route('/view/<rat_id>')
def rat_page(rat_id: str):
    """Rat page - shows regions for a specific rat"""
    regions = get_rat_regions(rat_id)
    metadata = get_rat_metadata(rat_id)
    if not regions:
        return render_template('error.html', message=f"Rat {rat_id} not found")
    return render_template('rat.html', rat_id=rat_id, regions=regions, metadata=metadata)

@app.route('/view/<rat_id>/<slice_name>')
def slice_page(rat_id: str, slice_name: str):
    """Slice page - shows subregions for a specific rat and slice"""
    subregions = get_rat_subregions(rat_id, slice_name)
    metadata = get_rat_metadata(rat_id)

    if not subregions:
        return render_template('error.html', message=f"Slice {slice_name} not found for rat {rat_id}")

    return render_template('rat.html', 
                         rat_id=rat_id, 
                         regions=subregions, 
                         metadata=metadata,
                         slice_name=slice_name)


@app.route('/view/<rat_id>/<slice_name>/<region>')
def region_page(rat_id: str, slice_name: str, region: str):
    """Region page - shows image, segmentation, and feature map"""

    # Validate image URL path
    image_url_path = validate_safe_path(Config.DATA_DIR, rat_id, slice_name, region, "th.tif")
    if not image_url_path:
        return render_template('error.html', message=f"ROI image not found for {rat_id}/{region}")

    # Convert TIF to JPG for browser compatibility
    image_filename = f"{rat_id}_{slice_name}_{region}_img.jpg"
    image_filename_result = convert_tif_to_jpg_and_save(image_url_path, image_filename, lumin_scale=2.0)
    if not image_filename_result:
        return render_template('error.html', message=f"Failed to convert image for {rat_id}/{region}")
    
    # Check for cached segmentation and convert if exists
    seg_cache_path = get_cached_segmentation_path(rat_id, slice_name, region)
    has_segmentation = seg_cache_path and seg_cache_path.exists()
    segmentation_filename = None
    if has_segmentation:
        seg_filename = f"{rat_id}_{slice_name}_{region}_seg.jpg"
        segmentation_filename = convert_tif_to_jpg_and_save(seg_cache_path, seg_filename, lumin_scale=5.0)
    
    # Check for cached feature map and convert if exists
    feature_cache_path = get_cached_feature_map_path(rat_id, slice_name, region)
    has_feature_map = feature_cache_path and feature_cache_path.exists()
    feature_map_filename = None
    if has_feature_map:
        feature_filename = f"{rat_id}_{slice_name}_{region}_features.jpg"
        feature_map_filename = convert_tif_to_jpg_and_save(feature_cache_path, feature_filename, lumin_scale=15.0)
    
   
    # Get directory paths - resolve to absolute paths and simplify
    image_dir_path = str(image_url_path.parent.resolve()).replace('\\', '/') if image_url_path else None
    seg_dir_path = str(seg_cache_path.parent.resolve()).replace('\\', '/') if seg_cache_path else None
    feature_dir_path = str(feature_cache_path.parent.resolve()).replace('\\', '/') if feature_cache_path else None
    
    return render_template('region.html', 
                         rat_id=rat_id, 
                         slice_name=slice_name,
                         region=region,
                         image_filename=image_filename_result,
                         segmentation_filename=segmentation_filename,
                         feature_map_filename=feature_map_filename,
                         has_segmentation=has_segmentation,
                         has_feature_map=has_feature_map,
                         image_dir_path=image_dir_path,
                         seg_dir_path=seg_dir_path,
                         feature_dir_path=feature_dir_path)


@app.route('/dataset_statistics')
def dataset_statistics_page():
    """Dataset Statistics page - shows tracing checker features and dataset distribution"""
    data = get_dataset_statistics()
    return render_template('dataset_statistics.html', data=data)

@app.route('/compare')
def compare_page():
    """Group-based comparison page - allows creating multiple groups for comparison"""
    if request.method == 'GET':
        rats = scan_available_rats()
        regions = get_directories_with_tif_images(Config.DATA_DIR)
        return render_template('compare.html', rats=rats, regions=regions)

@app.route('/experiment/<experiment_id>')
def experiment_page(experiment_id : str):
    """Group-based comparison page - allows creating multiple groups for comparison"""
    
    comparison_results_data = get_comparison_data(experiment_id)
    return render_template('experiment.html', data=comparison_results_data)

@app.route('/api/rats')
def api_rats():
    """JSON API endpoint for getting available rats"""
    rats = scan_available_rats()
    return jsonify(rats)

@app.route('/api/rats/<rat_id>/regions')
def api_regions(rat_id: str):
    """JSON API endpoint for getting regions for a rat"""
    regions = get_rat_regions(rat_id)
    return jsonify(regions)

# Convert form data to RatGroup objects
def construct_RatGroup(json_group):
    group_num = json_group.get('groupNum')
    group_name = json_group.get('groupName', f"Group {group_num}")
    rats = json_group.get('rats', [])
    regions = json_group.get('regions', [])
    if rats == ['ALL_RATS']: rats = ALL_RATS
    if regions == ['ALL_REGIONS']: regions = ALL_REGIONS
    return RatGroup(rats=rats, regions=regions, group_name=group_name)

@app.route('/api/compare/groups', methods=['POST'])
def api_compare_groups():
    """API endpoint for group-based comparison"""
    try:
        data = request.get_json()
        if not data or 'groups' not in data:
            return jsonify({'error': 'Invalid request format'}), 400
        
        groups = data['groups']
        if len(groups) < 2:
            return jsonify({'error': 'At least 2 groups required for comparison'}), 400

        # Extract experiment metadata
        experiment_name = data.get('experiment_name', '')
        experimenter_name = data.get('experimenter_name', '')

        if not Config.USED_SEGMENTATION_MODEL_PATH.exists():
            return jsonify({'error': f'Segmentation model not found at {Config.USED_SEGMENTATION_MODEL_PATH}. You should train a model before doing this step. Please check your config.json.'}), 500

        rat_groups = [construct_RatGroup(group_data) for group_data in groups]

        exp_id = run_quantification(rat_groups, experiment_name, experimenter_name)
        
        # Read the actual experiment data from the JSON file
        experiment_data = {}
        statistics_summary = []
        visualizations_summary = []
        tests_summary = []
        
        try:
            experiment_data_path = Config.EXPERIMENTS_DIR / str(exp_id) / "data.json"
            if experiment_data_path.exists():
                with open(experiment_data_path, 'r') as f:
                    experiment_data = json.load(f)
                
                # Extract statistics from groups
                groups_data = experiment_data.get('groups', [])
                total_rois = sum(len(group.get('roi_associations', {})) for group in groups_data)
                
                # Generate statistics summary with key metrics
                group_summaries = []
                for group in groups_data:
                    group_name = group.get('name', 'Unknown')
                    avg_density = group.get('average_density', 0)
                    best_model = group.get('best_model_name', 'N/A')
                    group_summaries.append(
                        f"{group_name}: density={avg_density:.4f}, model={best_model}"
                    )
                
                statistics_summary.append(
                    f"{len(groups_data)} groups analyzed ({total_rois} ROIs). " +
                    " | ".join(group_summaries)
                )
                
                # Generate visualizations summary
                visualizations_summary.append(
                    f"Inference results and model performance graphs generated. "
                    f"View detailed visualizations in the full results page."
                )
                
                # Generate tests summary
                if len(groups_data) >= 2:
                    tests_summary.append(
                        f"Statistical analysis completed for {len(groups_data)} groups. "
                        f"Bootstrap confidence intervals calculated. View detailed tests in full results."
                    )
                else:
                    tests_summary.append(
                        "Statistical tests require at least 2 groups for comparison."
                    )
            else:
                logger.warning(f"Experiment data file not found: {experiment_data_path}")
                statistics_summary.append("Experiment data file not found. Please check the experiment directory.")
                visualizations_summary.append("Visualizations will be available after data processing completes.")
                tests_summary.append("Statistical tests will be available after data processing completes.")
        except Exception as e:
            logger.error(f"Error reading experiment data: {e}")
            statistics_summary.append(f"Error reading experiment data: {str(e)}")
            visualizations_summary.append("Visualizations may be available in the full results page.")
            tests_summary.append("Statistical tests may be available in the full results page.")
        
        result = {
            'status': 'success',
            'message': 'Comparison completed successfully',
            'experiment_id' : exp_id,
            'groups_analyzed': len(rat_groups),
            'summary': {
                'total_rats': sum(len(g.rats) if g.rats != ALL_RATS else len(scan_available_rats()) for g in rat_groups),
                'total_regions': sum(len(g.regions) if g.regions != ALL_REGIONS else len(Config.PREDEFINED_REGIONS) for g in rat_groups)
            },
            'results': {
                'statistics': ' | '.join(statistics_summary) if statistics_summary else 'No statistics available',
                'visualizations': ' | '.join(visualizations_summary) if visualizations_summary else 'Visualizations generated',
                'tests': ' | '.join(tests_summary) if tests_summary else 'Statistical tests completed'
            },
            'rat_groups_created': [g.group_name for g in rat_groups]
        }
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in group comparison: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        return jsonify({'error': f'Comparison failed: {str(e)}'}), 500


@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', message="Internal server error"), 500


# Application startup
def create_app():
    """Application factory function"""
    
    if not Config.DATA_DIR.exists():
        logger.error(f"Data directory does not exist: {Config.DATA_DIR}")
    
    if not Config.CACHE_DIR.exists():
        logger.error(f"Cache directory does not exist: {Config.CACHE_DIR}")
    
    logger.info("Rat Brain GUI application initialized")
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5001)
