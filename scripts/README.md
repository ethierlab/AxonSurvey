# AxonSurvey Workflow Scripts

This directory contains user-friendly Python scripts for executing the complete AxonSurvey workflow. These scripts are designed to be easy to use for non-technical users while providing flexibility for advanced users.

## Overview

The AxonSurvey workflow consists of the following main steps:

1. **Setup Folder Structure** - Organize your raw full-size scans into an Images Dataset directory structure
2. **Add Images** - Place your raw image files in the appropriate folders within the Images Dataset
3. **Sample Data** - Create Tracings Datasets (training and test sets) by sampling patches from your Images Dataset
4. **Manual Tracing** - Label the sampled patches in your Tracings Datasets by tracing axons (using NeuronJ or similar tools)
5. **Train Model** - Train a neural network on your Tracings Datasets
6. **Run Inference** - Apply the trained model to all images in your Images Dataset for segmentation
7. **Analyze Results** - Use the Axon Survey GUI to analyze, compare, and download results

## Scripts

### 1. `setup_folder_structure.py`

Creates the directory structure needed for your Images Dataset (raw full-size rat brain scans).

**Basic Usage:**
```bash
python scripts/1-setup_folder_structure.py --rats rat301,rat302 --bregmas b516,b468 --regions contra_inner,contra_outer --output ./data/project_scans
```

**Using a Configuration File:**
```bash
# First, create an example config
python scripts/1-setup_folder_structure.py --create-example-config

# Edit the generated example_config.json, then:
python scripts/1-setup_folder_structure.py --config example_config.json
```

**Configuration File Format:**
```json
{
    "rat_list": ["rat301", "rat302"],
    "bregma_dict": {
        "rat301": ["b516", "b468"],
        "rat302": ["b252"]
    },
    "subregion_list": ["contra_inner", "contra_outer", "ipsi_inner", "ipsi_outer"],
    "base_path": "./data/project_scans"
}
```

**Output Structure:**
```
data/project_scans/
├── rat301/
│   ├── b516/
│   │   ├── contra_inner/
│   │   ├── contra_outer/
│   │   ├── ipsi_inner/
│   │   └── ipsi_outer/
│   └── b468/
│       └── ...
└── rat302/
    └── ...
```

**Minimal Test (fast execution):**
```bash
python scripts/1-setup_folder_structure.py --rats test_rat --bregmas b0 --regions test_region --output ./data/test_scans
```

### 2. `sample_data.py`

Samples image patches from your Images Dataset to create Tracings Datasets (training and test sets).

**Random Sampling (Recommended for beginners):**
```bash
# Create training dataset
python scripts/2-sample_data.py --random --input ./data/project_scans --output ./data/tracings/train --size 200 --patch-size 128

# Create test dataset
python scripts/2-sample_data.py --random --input ./data/project_scans --output ./data/tracings/test --size 100 --patch-size 128
```

**Neural Network-Based Sampling (Future feature):**
```bash
python scripts/2-sample_data.py --nn --model-path ./data/trained_models/default_model.pth --input ./data/project_scans --output ./data/tracings/train --size 200 --patch-size 128
```

**Parameters:**
- `--random` or `--nn`: Sampling strategy
- `--input`: Directory containing your Images Dataset (raw full-size scans)
- `--output`: Directory where the Tracings Dataset will be created (must be empty)
- `--size`: Number of samples to create
- `--patch-size`: Size of each patch (assumes square, e.g., 128 = 128x128)
- `--channel`: Channel name to sample from (default: "th")
- `--no-stratify`: Disable region stratification (sample proportionally to area)
- `--test-fake-tracings`: Generate fake random tracings for testing the training pipeline without manual labeling

**Output Structure:**
```
data/tracings/train/
├── img_0001/
│   ├── img.tif
│   ├── outer_mask.tif
│   └── info.txt
├── img_0002/
│   └── ...
└── ...
```

**Minimal Test (generate fake tracings for train and test):**
```bash
python scripts/2-sample_data.py --random --input ./data/project_scans --output ./data/tracings/dummy_train --size 50 --patch-size 128 --test-fake-tracings
python scripts/2-sample_data.py --random --input ./data/project_scans --output ./data/tracings/dummy_test --size 20 --patch-size 128 --test-fake-tracings
```

### 3. `train_model.py`

Trains a UNet neural network model for axon segmentation using your Tracings Datasets.

**Basic Usage:**
```bash
python scripts/3-train_model.py --epochs 50
```

**Advanced Usage:**
```bash
python scripts/3-train_model.py \
    --train-dir ./data/tracings/train \
    --test-dir ./data/tracings/test \
    --output ./data/trained_models/default_model.pth \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --input-size 128 \
    --display-epochs 5
```

**Parameters:**
- `--train-dir`: Directory containing training Tracings Dataset (default: ./data/tracings/train)
- `--test-dir`: Directory containing test Tracings Dataset (default: ./data/tracings/test)
- `--output`: Path where trained model will be saved (.pth file)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 32)
- `--learning-rate`: Learning rate (default: 0.0001)
- `--input-size`: Input image size (default: 128)
- `--display-epochs`: Display progress every N epochs (default: 5)
- `--no-scheduler`: Disable learning rate scheduler

**Training Progress:**
- The script automatically checks your datasets before training
- Training loss and validation metrics are displayed periodically
- Loss curves are automatically generated and displayed
- The model is saved automatically when training completes

**Minimal Test (fast execution, minimal resources):**
```bash
python scripts/3-train_model.py --train-dir ./data/tracings/dummy_train --test-dir ./data/tracings/dummy_test --output ./data/trained_models/test_model.pth --epochs 1 --batch-size 10
```

### 4. `run_inference.py`

Applies a trained model to all images in your Images Dataset for segmentation.

**Basic Usage:**
```bash
python scripts/4-run_inference.py --model ./data/trained_models/default_model.pth --input ./data/project_scans --output ./data/segmented_images
```

**Advanced Usage:**
```bash
python scripts/4-run_inference.py \
    --model ./data/trained_models/default_model.pth \
    --input ./data/project_scans \
    --output ./data/segmented_images \
    --input-size 128 \
    --channel th
```

**Parameters:**
- `--model`: Path to trained model (.pth file)
- `--input`: Directory containing your Images Dataset
- `--output`: Directory where segmented images will be saved
- `--input-size`: Input size for the model (must match training size, default: 128)
- `--channel`: Channel name to use (default: "th")

**Output Structure:**
```
data/segmented_images/
├── rat301/
│   ├── b516/
│   │   ├── contra_inner/
│   │   │   └── segmentation.tif
│   │   └── ...
│   └── ...
└── ...
```

**Minimal Test (fast execution, minimal resources):**
```bash
python scripts/4-run_inference.py --model ./data/trained_models/test_model.pth --input ./data/test_scans --output ./data/test_segmented --input-size 32
```

## Complete Workflow Example

Here's a complete example of using all scripts in sequence:

```bash
# Step 1: Create folder structure
python scripts/1-setup_folder_structure.py \
    --rats rat301,rat302 \
    --bregmas b516,b468 \
    --regions contra_inner,contra_outer,ipsi_inner,ipsi_outer \
    --output ./data/project_scans

# Step 2: Add your images manually to the created folders
# (Place .tif files in each region folder)

# Step 3: Create training Tracings Dataset
python scripts/2-sample_data.py \
    --random \
    --input ./data/project_scans \
    --output ./data/tracings/train \
    --size 200 \
    --patch-size 128

# Step 4: Create test Tracings Dataset
python scripts/2-sample_data.py \
    --random \
    --input ./data/project_scans \
    --output ./data/tracings/test \
    --size 100 \
    --patch-size 128

# Step 5: Manually trace axons in the sampled patches (Tracings Dataset)
# (Use NeuronJ or similar tool to create tracings.tif files)

# Step 6: Train the model
python scripts/3-train_model.py --epochs 50

# Step 7: Run inference on all images in your Images Dataset
python scripts/4-run_inference.py \
    --model ./data/trained_models/default_model.pth \
    --input ./data/project_scans \
    --output ./data/segmented_images

# Step 8: Analyze results using the GUI
cd gui
python gui/app.py
# Then navigate to http://localhost:5001 in your browser
```

## Adding Images

After creating the folder structure, you need to add your raw full-size scans to the Images Dataset:

1. **Image Format**: Use `.tif` or `.tiff` format
2. **Naming Convention**: 
   - For single-channel images: name the file `{channel}.tif` (e.g., `th.tif`, `dbh.tif`)
   - For multi-channel images: the script will handle channel splitting automatically
3. **Placement**: Place images in the appropriate region folders:
   ```
   data/project_scans/rat301/b516/contra_inner/th.tif
   ```

**Video Tutorial**: For visual guidance on organizing images, see: [YouTube Tutorial Link](https://www.youtube.com/watch?v=5ao4zGMchgY)

## Manual Tracing

After sampling, you need to manually trace axons in the sampled patches (Tracings Dataset):

1. **Tool**: Use NeuronJ (ImageJ plugin) or similar tracing tools
2. **Output**: Save tracings as `tracings.tif` in each sample folder:
   ```
   data/tracings/train/img_0001/tracings.tif
   ```
3. **Format**: Tracings should be binary masks (white = axon, black = background)

**Video Tutorial**: For guidance on tracing axons, see: [Make a video] (link to be added)

## Using the Axon Survey GUI

After running inference, you can analyze results using the web-based GUI:

1. **Start the GUI**:
   ```bash
   cd gui
   python gui/app.py
   ```

2. **Access the Interface**: Open your browser and navigate to `http://localhost:5001`

3. **Features**:
   - Browse segmented images by rat and region
   - Compare different regions and rats
   - Download comparison data
   - Run statistical analyses

4. **Compare Page**: Use the compare page to:
   - Create groups of rats/regions for comparison
   - Generate statistical comparisons
   - Download results for further analysis

## Troubleshooting

### Common Issues

**"Output directory is not empty"**
- The output directory for sampling must be empty or not exist
- Solution: Use a new directory name or remove existing files

**"Dataset has no labeled data"**
- You need to add tracings to your sampled images
- Solution: Manually trace axons and save as `tracings.tif` in each sample folder

**"CUDA not available"**
- Training will run on CPU (slower but functional)
- Solution: Install CUDA and PyTorch with GPU support if you have a compatible GPU

**"Model file not found"**
- Check that the model path is correct
- Solution: Ensure the model was trained successfully and the path is absolute or relative to your current directory

### Getting Help

- Check the script help: `python scripts/script_name.py --help`
- Review the error messages carefully - they often indicate what's wrong
- Ensure all required dependencies are installed: `pip install -r requirements.txt`

## Advanced Usage

### Custom Sampling Strategies

For advanced users, you can create custom sampling strategies by modifying `src/dataprep/SamplingStrategies.py` and implementing new strategy classes.

### Model Customization

You can customize the model architecture by modifying `src/NNs/Unet.py` or creating new model classes.

### Batch Processing

For large datasets, consider processing in batches or using the GUI's batch processing features.

## Next Steps

After completing the workflow:

1. **Evaluate Model Performance**: Check training metrics and validation results
2. **Refine Training**: Adjust hyperparameters or add more training data if needed
3. **Analyze Results**: Use the GUI to explore segmented images and compare regions
4. **Export Data**: Download comparison data for statistical analysis in external tools

## Additional Resources

- **Project Documentation**: See `docs/` folder for detailed API documentation
- **Example Notebooks**: See `workflows/` folder for Jupyter notebook examples
- **Source Code**: See `src/` folder for implementation details

