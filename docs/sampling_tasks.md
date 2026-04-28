# Sampling System Tasks

This document contains a list of dense, concrete tasks to finalize and improve the sampling architecture (`SampleSaver.py` and `SamplingStrategies.py`).

## High Priority
- [ ] **Test with Real Data**: Validate the new `SampleSaver` and `SamplingStrategy` architecture with real datasets to ensure end-to-end functionality.
- [ ] **Implement Multi-Group Stratification**: Currently, providing multiple `RatGroup` objects with `stratify_group=True` emits a `UserWarning` (NotImplementedWarning). Implement the logic to sample equally (or proportionally) across multiple experimental groups (e.g., control vs. treatment).
- [ ] **Support Existing Labeled Datasets**: The `existing_labeled_dataset` parameter in `SampleSaver.__init__` is currently ignored. Implement the logic to reuse existing labeled samples (e.g., by restricting the sampling strategy to SRS and avoiding already sampled regions).

## Medium Priority
- [ ] **Delete Legacy Code**: `DataSampler.py` is now obsolete. Verify no scripts depend on it and safely delete the file.
- [ ] **Implement `suggest_n()`**: The base `SamplingStrategy.suggest_n()` method currently returns `-1`. Implement meaningful logic to suggest an appropriate sample size based on the population and strategy.
- [x] **Clean Up Test Flags**: Remove or formalize the `test_fake_tracings` flag in `SampleSaver.create_dataset()`.

## Low Priority
- [ ] **Review Area-Proportional Sampling**: Ensure that `stratify_regions=False` correctly samples proportionally to the area of each region across all edge cases.
- [ ] **Refine Error Handling**: Ensure clear error messages when `sample_dimensions` are larger than the available image size (currently just skips the image with a print statement).
