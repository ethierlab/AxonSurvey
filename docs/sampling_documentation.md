# Sampling System Architecture

## Overview
The sampling architecture separates concerns into two main components:
1. **`SamplingStrategy`** (in `src/dataprep/SamplingStrategies.py`): Handles sampling logic, manages data reading, and determines which patches to sample.
2. **`SampleSaver`** (in `src/dataprep/SampleSaver.py`): Handles saving logic, loads the population based on the strategy, and saves the sampled results to disk.

*(Note: The legacy `DataSampler.py` has been deprecated in favor of this modular approach.)*

## Components Detail

### 1. `SamplingStrategy`
- **Responsibilities**:
  - Initializes `DataReader` with the project directory.
  - Manages `channel`, `sample_dimensions`, and `groups`.
  - Computes population structure and determines which indices to sample.
- **RatGroup Integration**:
  - `groups` parameter accepts `RatGroup` objects (single or list).
  - **Default**: Single `RatGroup` with `ALL_RATS` and `ALL_REGIONS` (samples from the entire dataset).
  - **Subset**: If a single `RatGroup` specifies a subset, only paths matching both rat and region criteria are included.
  - **Multiple Groups**: Defines separate sampling strata (e.g., control vs. treatment).
- **Subclasses**:
  - `SRS` (Simple Random Sampling): Randomly selects unique indices, with options to stratify by region (`stratify_regions=True` samples equally from each region, `False` samples proportionally to area).

### 2. `SampleSaver`
- **Responsibilities**:
  - Validates the target directory (must be empty).
  - Receives a `SamplingStrategy` instance.
  - Loads the population of images, masks, and starting points.
  - Calls the strategy to sample indices.
  - Saves the resulting images, masks, and metadata (`info.txt`) to disk.
- **Population Loading**:
  - Creates sliding window patches of size `sample_dimensions`.
  - Filters patches to keep only those with >50% valid mask coverage (outer mask).

### Path Filtering Logic
The system filters available paths based on the provided `RatGroup` objects:
```python
for rat_group in groups:
    filtered_paths = [
        path for path in all_paths
        if rat_group.include_rat(dr.get_rat(path)) and 
           rat_group.include_region(dr.get_region(path))
    ]
    population_paths.append(filtered_paths)
```
This relies on `DataReader.get_rat(path)` and `DataReader.get_region(path)` to extract metadata from the directory structure.
