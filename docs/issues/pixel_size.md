# Pixel Size and Resolution Handling Issues

## Overview
The codebase currently assumes a uniform pixel size (defaulting to 1.0) and lacks mechanisms to handle images with different resolutions. This leads to critical errors in density, length, and area calculations when dealing with varying pixel sizes across different images.

## Key Issues Found

1. **Missing Metadata**: `info.txt` stores starting points and file paths but no pixel size or resolution information.
2. **Hardcoded Defaults**: Functions like `get_trace_density()` and `get_mean_axon_length()` default to `pixel_length=1.0` in `src/utils/traceProps.py`.
3. **Incorrect Density Calculations**: In `src/evaluation/Evaluator.py` and `src/image_extractors/TracerExtractor.py`, ground truth functions are called without providing `pixel_length`, causing density values to be off by factors of up to 10,000x for 0.01 micron pixels.
4. **Hardcoded Values in Density Estimation**: `estimate_axon_density` in `src/imgproc/density_estimation.py` hardcodes `pixel_to_micron=0.1` and `image_width=400`.
5. **Flawed Shape Validation**: `ImageLoader.py` enforces uniform pixel dimensions (e.g., 128x128) but ignores physical resolution, which hides resolution mismatches.
6. **Invalid Area Comparisons**: `DataReader.py` computes areas in pixels rather than physical units, making comparisons across different resolutions invalid.

## Proposed Solutions

1. **Metadata Storage**: Transition to JSON-based metadata (`meta.json`) to store `pixel_size_micron` alongside other image properties, or extend `info.txt` to include this information.
2. **Metadata Integration**: Update `ImageLoader` to read and store pixel sizes per image.
3. **Parameter Propagation**: Update `Evaluator`, `TracerExtractor`, and `DataSampler` to pass and save `pixel_length` correctly.
4. **Validation**: Implement warnings or errors when mixing resolutions without normalization.
5. **Backward Compatibility**: Provide a migration script or default values (with warnings) for existing datasets lacking pixel size metadata.

## Criticality
- **High**: Density and length calculations are fundamentally incorrect when pixel sizes differ.
- **Medium**: Area comparisons are invalid without pixel size conversion.
- **Low**: Shape validation may hide resolution issues (but prevents dimension mismatches).
