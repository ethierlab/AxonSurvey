# Work in progress

import os
import warnings
import numpy as np
from abc import abstractmethod

from ..dataprep.DataReader import DataReader
from ..experiments.RatGroup import RatGroup, ALL_RATS, ALL_REGIONS

class SamplingStrategy:
    """
    Base class for sampling strategies.
    Subclasses should implement the sample_indices method.
    """
    def __init__(self, project_raw_image_dir, groups, channel, sample_dimensions, stratify_regions=True, stratify_group=True):

        """
        The data passed to the sampling strategy must be able to represent enough information to do every kind of sampling. 
        
        Args:
            project_raw_image_dir (str): Root directory containing the dataset.
            groups (RatGroup, list of RatGroup, or None): RatGroup object(s) defining which rats/regions to sample from.
                - Single RatGroup: Samples from that group's subset
                - List of RatGroup: Multiple groups for stratification (if stratify_group=True, emits NotImplementedWarning)
                - None: Defaults to single RatGroup with ALL_RATS and ALL_REGIONS
            channel (str): Channel name to sample from each image.
            sample_dimensions (tuple): (height, width) of each sample patch.
            stratify_group (bool): If True, sample equally from each group. If False, sample proportionally to the area of each group.
            stratify_regions (bool): If True, sample equally from each region. If False, sample proportionally to the area of each region.
        
        By default, both are True.

        Sampling strategies may or may not require the image information, but it"s always passed and it's used if needed.
        """

        if not (os.path.exists(project_raw_image_dir)): raise ValueError(f"read directory {project_raw_image_dir} doesn't exist")
        self.dr = DataReader(project_raw_image_dir)
        if not (self.dr.read_dir_is_valid()): raise ValueError(f"read directory {project_raw_image_dir} not valid")

        self.root_read_dir = project_raw_image_dir
        self.channel = channel

        self.sample_dimensions = sample_dimensions
        self.sample_area = sample_dimensions[0] * sample_dimensions[1]

        # Normalize groups to always be a list of RatGroup objects
        if groups is None:
            groups = [RatGroup(rats=ALL_RATS, regions=ALL_REGIONS)]
        elif isinstance(groups, RatGroup):
            groups = [groups]
        elif isinstance(groups, list):
            if not all(isinstance(g, RatGroup) for g in groups):
                raise TypeError("If groups is a list, all elements must be RatGroup objects")
        else:
            raise TypeError("groups must be a RatGroup, list of RatGroup, or None")
        
        # Warn if multiple groups with stratify_group=True (not yet implemented)
        if len(groups) > 1 and stratify_group:
            warnings.warn(
                "Multiple groups provided with stratify_group=True. Multi-group stratification is not yet implemented.",
                UserWarning
            )
        
        self.groups = groups
        self.stratify_regions = stratify_regions
        self.stratify_group = stratify_group

        self.all_subregion_folders = self.dr.get_paths()

    def get_groups(self):
        """Return the groups used for sampling."""
        return self.groups

    def get_group_count(self): 
        """Return the number of groups."""
        if self.groups is None:
            return 1
        else: return len(self.groups)

    def set_population(self, population):
        """Set the population and compute group and region counts."""
        self.population = population
        self.group_counts, self.region_count = self.get_group_and_region_counts_counts()
        
    def get_group_and_region_counts_counts(self):
        """Return group counts and region counts for the population."""
        region_counts = []
        for group in self.population: region_counts.append([len(region) for region in group])
        group_counts = [sum(group) for group in region_counts]
        return group_counts, region_counts


    @abstractmethod
    def sample_indices(self, n):
        """
        Return indices of the sampled items.

        Args:
            n (int): Number of items to sample.

        Returns:
            list: Nested list structure matching population structure.
                 Format: [group_indices, ...] where each group_indices is [region_indices, ...]
                 and each region_indices is a numpy array of selected patch indices.
        """
        raise NotImplementedError("Subclasses must implement sample_indices.")
    
    def distribute_remainder(self, exact_quantities, total_target):
        """
        Distributes the remainder of fractional quantities to reach the total_target.
        Uses the largest remainder method.
        """
        floored = np.floor(exact_quantities).astype(int)
        remainder = int(total_target - np.sum(floored))
        
        if remainder > 0:
            fractions = exact_quantities - floored
            # argsort sorts ascending, so we take the last 'remainder' elements
            indices = np.argsort(fractions)[-remainder:]
            for idx in indices:
                floored[idx] += 1
                
        return floored

    def suggest_n(self):
        """Suggest sample size n (default -1)."""
        return -1

class SRS(SamplingStrategy):
    """
    Simple Random Sampling (SRS): randomly selects sample_size unique indices.
    """
    def __init__(self, project_raw_image_dir, groups, channel, sample_dimensions, stratify_regions=True, stratify_group=True, **kwargs):
        """Initialize SRS sampling strategy."""
        super().__init__(project_raw_image_dir, groups, channel, sample_dimensions, stratify_regions=stratify_regions, stratify_group=stratify_group, **kwargs)

    def sample_indices(self, n):
        """Sample indices for each group and region."""
        num_groups = self.get_group_count()
        group_exact = np.full(num_groups, n / num_groups)
        group_targets = self.distribute_remainder(group_exact, n)
        
        selected = []
        for group, region_counts, group_target in zip(self.population, self.region_count, group_targets):

            if self.stratify_regions: sampling_proportions = np.full(len(group,), 1/len(group))
            else : sampling_proportions = (np.array(region_counts) / np.sum(region_counts)) # proportionnal to img size

            exact_quantities = group_target * sampling_proportions
            sampling_quantities = self.distribute_remainder(exact_quantities, group_target)
            
            selected_indices = [self.get_n_random_indices_in_list(quantity, region) for region, quantity in zip(group, sampling_quantities)]
            selected.append(selected_indices)
        return selected


    def get_n_random_indices_in_list(self, n, elements):
        """Return n unique random indices from the elements list."""
        if n > len(elements):
            raise ValueError("n cannot be greater than the number of elements.")
        rng = np.random.default_rng()
        a = rng.choice(len(elements), size=n, replace=False)
        return a

