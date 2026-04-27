# Scripts Summary

This document provides a quick overview of all available workflow scripts in the `scripts/` directory.

## Overview

The AxonSurvey workflow scripts transform complex Jupyter notebook workflows into simple, executable Python scripts that non-technical users can run from the command line. Each script is self-contained, well-documented, and provides clear error messages.

## Scripts

### 1. `scripts/1-setup_folder_structure.py`
**Purpose**: Creates the directory structure for organizing rat brain image data.

**Key Features**:
- Creates nested folder structure: `rat/bregma/region/`
- Supports JSON configuration files for easy setup
- Can generate example configuration files
- Validates input parameters

**When to Use**: First step in the workflow, before adding images.

---

### 2. `scripts/2-sample_data.py`
**Purpose**: Samples image patches from project data to create training and test datasets.

**Key Features**:
- Random sampling (simple and recommended)
- Neural network-based sampling (future feature, currently falls back to random)
- Stratified sampling by region
- Configurable patch size and sample count
- Automatic mask generation

**When to Use**: After setting up folder structure and adding images, before manual tracing.

---

### 3. `scripts/3-train_model.py`
**Purpose**: Trains a UNet neural network model for axon segmentation.

**Key Features**:
- Automatic dataset validation
- Configurable training parameters (epochs, batch size, learning rate)
- Learning rate scheduling with warmup
- Progress monitoring and loss visualization
- Automatic model saving

**When to Use**: After manually tracing axons in sampled images.

---

### 4. `scripts/4-run_inference.py`
**Purpose**: Applies a trained model to all images in the project for segmentation.

**Key Features**:
- Processes all images in project structure
- Maintains directory structure in output
- Error handling for individual images
- Progress reporting

**When to Use**: After training a model, before analyzing results in the GUI.

---

## Workflow Order

```
1. scripts/1-setup_folder_structure.py
   ↓
2. [Manually add images to folders]
   ↓
3. scripts/2-sample_data.py (for training dataset)
   ↓
4. scripts/2-sample_data.py (for test dataset)
   ↓
5. [Manually trace axons in sampled images]
   ↓
6. scripts/3-train_model.py
   ↓
7. scripts/4-run_inference.py
   ↓
8. [Use Axon Survey GUI for analysis]
```

## Quick Reference

| Script | Input | Output | Key Parameters |
|--------|-------|--------|----------------|
| `scripts/1-setup_folder_structure.py` | Rat list, bregmas, regions | Folder structure | `--rats`, `--bregmas`, `--regions` |
| `scripts/2-sample_data.py` | Project images | Sampled dataset | `--random`, `--size`, `--patch-size` |
| `scripts/3-train_model.py` | Training/test datasets | Trained model (.pth) | `--train-dir`, `--test-dir`, `--epochs` |
| `scripts/4-run_inference.py` | Trained model, project images | Segmented images | `--model`, `--input`, `--output` |

## Documentation

For detailed documentation, see:
- **README.md**: Comprehensive guide with examples and troubleshooting
- **Script help**: Run any script with `--help` flag for usage information
- **Source code**: Each script includes detailed docstrings

## Requirements

All scripts require:
- Python 3.8+
- Project dependencies (install with `pip install -r requirements.txt`)
- Properly structured data directories

## Getting Started

1. Read the **README.md** for detailed instructions
2. Start with `scripts/1-setup_folder_structure.py` to create your directory structure
3. Follow the workflow order listed above
4. Use `--help` on any script for parameter information

## Support

For issues or questions:
- Check script error messages (they often indicate the solution)
- Review the README.md troubleshooting section
- Ensure all dependencies are installed correctly

