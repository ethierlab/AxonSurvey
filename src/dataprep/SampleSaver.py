# Work in progress

import os
import pathlib
import numpy as np
import warnings

from abc import abstractmethod
from skimage.util import view_as_windows

from ..dataprep.DataReader import DataReader
from ..dataprep.TracingChecker import TracingChecker
from ..utils.imageio import tif_to_numpy, numpy_to_tif
from ..experiments.RatGroup import RatGroup, ALL_RATS, ALL_REGIONS

from ..dataprep.SamplingStrategies import SamplingStrategy

class SampleSaver:
    """
    SampleSaver generates labeled sample datasets from a larger pool of unlabeled images.
    It supports lots of sampling methods, SRS, stratified or model-based

    This class supports:
      - Uniform or area-proportional sampling from subregions.
      - Stratification by region.
      - Reusing existing labeled samples if available.
      - Sampling random patches from large images, with configurable patch size.
      - Saving sampled images, masks, and metadata to a new dataset directory.

    Args:
        sample_dimensions (tuple): (height, width) of each sample patch.
        new_dataset_dir (str): Directory to save the new labeled dataset.
        unlabeled_dir (str): Directory containing the source images.
        channel (str): Channel name to sample from each image.
        existing_labeled_dataset (str or None): Path to an existing labeled dataset to reuse samples from.
        uniform_sampling (bool): If True, sample equally from each subregion; else, sample by area.
        stratify_by_region (bool): If True, stratify sampling by region.
    """

    # sampling_strategy has the information for unlabeled_dir, sample_dimensions, channel and groups for stratification
    # # uniform_sampling = True, stratify_by_region = True

    # In other words SampleSaver only handles the saving logic, not the sampling logic

    # Will ignore existing_labeled_dataset for now, a simple way of 
    # combining this with sampling_strategy might look like only allowing sampling_strategy to be SRS 

    
    def __init__(self, new_dataset_dir, sampling_strategy, existing_labeled_dataset=None):

        
        if (os.listdir(new_dataset_dir)): raise ValueError(f"new dataset directory {new_dataset_dir} is not empty. use new name or remove existing one manually")
        if not (os.path.exists(new_dataset_dir)): raise ValueError(f"new dataset directory {new_dataset_dir} doesn't exist")

        self.existing_labeled_dataset = existing_labeled_dataset
        if existing_labeled_dataset is not None:
            if not (os.path.exists(existing_labeled_dataset)): raise ValueError(f"existing dataset {existing_labeled_dataset} doesn't exist")
            self.tracing_checker = TracingChecker(existing_labeled_dataset)
            if not (self.tracing_checker.is_valid()): raise ValueError(f"existing dataset {existing_labeled_dataset} isn't valid")
            if (self.tracing_checker.get_labelled_ratio() == 0.0): raise ValueError(f"existing dataset {existing_labeled_dataset} has no labelled data")
        
        self.root_write_dir = new_dataset_dir


    
        self.sampling_strategy = sampling_strategy
        self.population_paths = self.get_population_path_structure()
        print(self.population_paths)
        self.population, self.masks, self.starting_points = self.load_population(self.population_paths)
        self.sampling_strategy.set_population(self.population)

        # creates really basic tracings just to test things out
        self.test_fake_tracings = False
    
    def get_population_path_structure(self):
        """Gets population path structure based on sampling strategy RatGroups."""
        groups = self.sampling_strategy.get_groups()
        dr = self.sampling_strategy.dr
        all_paths = self.sampling_strategy.all_subregion_folders
        
        # Filter paths based on RatGroup criteria
        population_paths = []
        for rat_group in groups:
            filtered_paths = [
                path for path in all_paths
                if rat_group.include_rat(dr.get_rat(path)) and 
                   rat_group.include_region(dr.get_region(path))
            ]
            population_paths.append(filtered_paths)
        
        return population_paths
    
    def load_population(self, population_paths):
        """
        Loads population images, masks, and starting points from paths.
        
        For each image, creates patches using sliding windows and filters patches
        with >50% mask coverage (valid pixels).
        
        Args:
            population_paths: List of lists of paths, organized by group and region.
            
        Returns:
            tuple: (population, masks, starting_points)
                - population: List of groups, each containing lists of regions, each containing lists of image patches
                - masks: Same structure as population, containing boolean masks for each patch
                - starting_points: Same structure, containing (row, col) starting points for each patch
        """
        outer_acceptable_ratio = 0.5  # Minimum 50% mask coverage required
        population = []
        pop_masks = []
        pop_points = []
        
        dr = self.sampling_strategy.dr
        channel = self.sampling_strategy.channel
        sample_dimensions = self.sampling_strategy.sample_dimensions
        sample_area = self.sampling_strategy.sample_area

        for group in population_paths:
            regions_im, regions_ma, regions_po = [], [], []
            for path in group:
                img_path = os.path.join(path, f'{channel}.tif')
                print(f"opening image {img_path}")
                
                # Load image and create sliding window patches
                full_image = tif_to_numpy(img_path, output_dims=2)
                
                # Check if image is large enough for sample dimensions
                if sample_dimensions[0] > full_image.shape[0] or sample_dimensions[1] > full_image.shape[1]:
                    print(f"WARNING: skipping image at {img_path} because too small for sample dimensions")
                    regions_im.append([])
                    regions_ma.append([])
                    regions_po.append([])
                    continue
                
                # Create sliding windows (non-overlapping patches)
                images_window = view_as_windows(full_image, window_shape=sample_dimensions, step=sample_dimensions)
                images = images_window.reshape(-1, sample_dimensions[0], sample_dimensions[1])
                
                # Create masks for each patch (valid pixels = non-zero)
                masks = np.array([dr.get_outer_mask(im) for im in images])
                
                # Create starting points for each patch
                points = np.empty(images_window.shape[:2], dtype=object)
                for i in range(points.shape[0]): 
                    for j in range(points.shape[1]):
                        points[i, j] = (i * sample_dimensions[0], j * sample_dimensions[1])
                points = points.reshape(-1)
                
                # Filter patches: keep only those with >50% mask coverage
                func = lambda mask: (np.sum(mask) / sample_area) > outer_acceptable_ratio
                valid_indices = [i for i, mask in enumerate(masks) if func(mask)]

                images = [images[i] for i in valid_indices]
                masks = [masks[i] for i in valid_indices]
                points = [points[i] for i in valid_indices]

                print(f"found {len(images)} valid patches (of {len(images_window.reshape(-1))} total), shape {sample_dimensions}")
 
                regions_im.append(images)
                regions_ma.append(masks)
                regions_po.append(points)

            population.append(regions_im)
            pop_masks.append(regions_ma)
            pop_points.append(regions_po)

        return population, pop_masks, pop_points
    
    def get_sample_from_groups_indices(self, indices):
        """
        Gets samples from group indices and returns formatted sample data.
        
        Args:
            indices: Nested list structure matching population structure.
                   Format: [group_indices, ...] where each group_indices is [region_indices, ...]
                   and each region_indices is a numpy array of selected patch indices.
        
        Returns:
            tuple: (images, masks, starting_points, original_paths)
        """
        sample = ([], [], [], [])
        for grp_indices, grp_ROI, grp_masks, grp_points, grp_paths in zip(
            indices, self.population, self.masks, self.starting_points, self.population_paths
        ):
            for reg_indices, reg_ROI, reg_masks, reg_points, reg_path in zip(
                grp_indices, grp_ROI, grp_masks, grp_points, grp_paths
            ):
                for sampled_index in reg_indices:
                    sample[0].append(reg_ROI[sampled_index])
                    sample[1].append(reg_masks[sampled_index])
                    sample[2].append(reg_points[sampled_index])
                    sample[3].append(reg_path)
        return sample
    
    def create_dataset(self, size):
        """
        Creates a labeled dataset of the given size and saves to disk.
        
        Args:
            size (int): Number of samples to create.
            
        Returns:
            tuple: (total_samples_created, path_counts_dict)
        """
        sample_number = 0
        samples = self.sample(size)
        path_counts = {}

        for img, mask, point, file in zip(*samples):
            sample_number += 1
            save_path = os.path.join(self.root_write_dir, f"img_{sample_number:04d}")
            os.makedirs(save_path, exist_ok=True)
            
            # Save uncompressed for direct viewing
            numpy_to_tif(img, os.path.join(save_path, "img.tif"), compression=False)
            numpy_to_tif(mask.astype(np.uint8) * 255, os.path.join(save_path, "outer_mask.tif"), compression=False)

            if self.test_fake_tracings:
                numpy_to_tif(
                    (np.random.rand(img.shape[0], img.shape[1]) > 0.95).astype(np.uint8) * 255,
                    os.path.join(save_path, "tracings.tif"),
                    compression=False
                )
            
            pth = pathlib.Path(file[len(self.sampling_strategy.root_read_dir):])
            file_parts = list(pth.parts)

            with open(os.path.join(save_path, "info.txt"), "w") as f:
                f.write(str(point))
                f.write('\n')
                f.write(file)
                f.write('\n')
                for part in file_parts: 
                    f.write(part)
                    f.write('\n')
                    
            path_counts[file] = path_counts.get(file, 0) + 1
            
        return sample_number, path_counts
    
    def sample(self, n):
        """
        Samples n patches from population using the sampling strategy.
        
        Args:
            n (int): Number of samples to draw.
            
        Returns:
            tuple: (images, masks, starting_points, original_paths)
        """
        sampled_population_indices = self.sampling_strategy.sample_indices(n)
        return self.get_sample_from_groups_indices(sampled_population_indices)