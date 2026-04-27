## Task: Two-Channel Axon Segmentation and Subtraction-Based Quantification

### Overview
We need to extend the current project—which supports segmentation of a simple one-dimensional image—to handle two-channel .tif inputs for axon quantification. The core requirement is to segment axons in both channels using separate models, then subtract the second channel’s segmented axons (after making them thicker programmatically) from the first channel’s segmentation. The quantification will measure axon density in the first channel minus the second.

### High-Level Goals
- Run segmentation on both channels of a two-channel .tif input, using different models per channel.
- Post-process the second channel’s segmentation by making axon components thicker.
- Subtract the thickened second-channel mask from the first-channel mask.
- Quantify density on the resulting subtraction mask.
- Make the pipeline flexible to select channels and models interchangeably, with clear documentation.

---

## Data and Inputs
- Input format: two-channel .tif images.
- Each channel originates from different data sources and requires a different segmentation model.

---

## Segmentation and Processing Logic

### 1) Segmentation
- Channel 1: Run Model A to obtain the axon segmentation mask (leave as-is).
- Channel 2: Run Model B to obtain the axon segmentation mask (different model due to different data/source).

### 2) Post-Processing (Channel 2 Only)
- After segmentation of Channel 2, make the axon components thicker programmatically.
- Approach: take all “1” pixels in the mask and apply the existing function in the project that thickens lines/components.
- Channel 1’s mask is not modified.

### 3) Subtraction
- Compute: Channel 1 mask MINUS thickened Channel 2 mask.
- There are no negative pixels: if a pixel is 0 in Channel 1 and 1 in Channel 2 (after thickening), it remains 0.
- Effective result: pixels left represent fibers present in Channel 1 that are not present in Channel 2, accounting for imperfect overlap by thickening Channel 2 before subtraction.

### 4) Quantification
- Perform axon density quantification on the subtraction result (i.e., density in Channel 1 minus Channel 2 after processing).
- This replaces the current single-channel density quantification when enabled.

---

## Implementation Plan

### Inference Pipeline Changes
- Add support to run a second model on the second channel of the input images.
- Introduce configuration to:
  - Provide a path to the second model (Model B).
  - Enable/disable the subtraction-based quantification mode.
  - Select which channel is treated as “first” and which as “second” (interchangeability).

### Parameterization
- Model paths: model_a_path (Channel 1), model_b_path (Channel 2).
- Mode flag: use_subtraction_quantification = true/false.
- Channel selection: parameters to map models to channels flexibly.

### Processing Steps (When Subtraction Mode Is Enabled)
1. Load two-channel .tif.
2. Segment Channel 1 with Model A → Mask1.
3. Segment Channel 2 with Model B → Mask2.
4. Apply the existing mask-thickening function to Mask2 → Mask2_thick.
5. Compute Subtraction: ResultMask = Mask1 minus Mask2_thick (no negatives).
6. Quantify density on ResultMask.

---

## Model and Training Considerations
- A second model is required for Channel 2.
- Verify the codebase supports training a model for the second channel with minimal changes.
- Prepare a second dataset for training Model B (requires its own sampling process).
- Ensure the code allows using Channel 1 or Channel 2 interchangeably during training and inference.

---

## Documentation Requirements
- Document how to:
  - Configure model paths for both channels.
  - Enable the subtraction-based quantification mode.
  - Select which model runs on which channel.
  - Apply and tune the mask-thickening step (reference the existing function).
- Provide examples for both the standard single-channel density mode and the new subtraction mode.

---

## Notes
- This is more complex than straightforward mask subtraction because fibers may not overlap perfectly between channels; thickening Channel 2’s mask addresses this before subtraction.
- The first channel remains unmodified; only the second channel’s mask is thickened prior to subtraction.
- We believe there is an existing function in the project to thicken mask components.

---

## To-Do Checklist
- Add second-model inference to the pipeline.
- Implement channel selection and interchangeability in code.
- Integrate the existing mask-thickening function for Channel 2.
- Implement subtraction and ensure non-negative result handling.
- Add configuration parameters and CLI/ API flags for the new mode.
- Update quantification to operate on the subtraction result.
- Validate end-to-end on example two-channel .tif inputs.
- Confirm training support for Model B; create and sample the second dataset.
- Write and publish documentation for setup, configuration, and usage.


-------





## Task Overview

The next task involves significant modifications and expansions. The goal is to produce a large markdown document that explains what needs to be done. The task will later be divided into smaller parts, but for now, here is the detailed requirement and the plan for implementation.

---

## Current Project Overview

- The current project allows segmentation of a simple one-dimensional image.
- Now, we need to handle input images in `.tiff` format that contain two channels.

---

## Required Modifications

### Data Processing

- **Channel Utilization:**
  - We need to take the axons from one channel, and subtract the other channel from that channel.
  - Each channel comes from different data sources, meaning they require **different segmentation models**.

- **Segmentation Workflow:**
  1. Run a segmentation model on the first channel.
  2. Run a different segmentation model on the second channel.
  3. Extract segmentation masks (fibers) from both channels.

- **Quantification:**
  - Instead of quantifying the density in just one channel, calculate the density from the first channel **minus** the second channel.

---

### Mask Subtraction and Processing

- **Challenge:** Direct mask subtraction is not sufficient because fibers might not overlap perfectly.
- **Solution:**
  - First channel mask: leave as is (thin fibers).
  - Second channel mask: **inflate** the fibers (make axons thicker). Use an existing function in the project that increases the thickness by turning components of the mask into “thicker” lines (run an algorithm that expands all one-valued pixels).
  - Subtract the **inflated** second channel mask from the first channel mask.
  - The subtraction result will contain only the fibers present in the first image but not overlapping with the expanded fibers in the second.

---

### Resulting Mask Characteristics

- **No Negative Pixels:** Any pixel that’s zero in the first channel or corresponds to the thicker (ones) in the second channel will remain zero in the result.
- **Effective Subtraction:** This will leave only actual fibers present in the first channel but absent in the second channel.

---

## Implementation Considerations

- **Inference Pipeline:**
  - Current pipeline should be adaptable to incorporate these changes.
- **Additional Model:**
  - A second segmentation model will need to be loaded and used for the second channel.
  - There should be a parameter or function to select this quantification approach instead of the existing one for flexibility.
- **Quantification Pipeline:**
  - Very similar to the current axon density quantification.
  - Only difference: run the second model on the second channel, then apply the described subtraction logic.
- **Code Flexibility:**
  - It should be easy to switch between using the first or second channel in the code.
- **Documentation:**
  - Detailed documentation is required to describe this new workflow and make it clear how to use either channel.

---

## Training a Second Model

- **Model Training:**
  - It’s necessary to check if the code is flexible and straightforward enough to allow training a model on the second channel.
- **Additional Dataset:**
  - A separate dataset must be sampled for the second channel model.
  - Code and documentation must support this interchangeability and dataset management.

---

## Summary of Action Points

1. **Update the inference pipeline** to support two-channel input and dual model segmentation.
2. **Implement mask inflation logic** for the second channel.
3. **Create the subtraction pipeline** that quantifies axon density as (first channel mask) minus (inflated second channel mask).
4. **Ensure flexibility** in code to use either channel and different models as needed.
5. **Document all changes** clearly for both users and future developers.
6. **Verify codebase** allows straightforward extension to new models, and sample a new dataset for the second channel.

---

*Further division and breakdown of tasks will follow as the next steps.*