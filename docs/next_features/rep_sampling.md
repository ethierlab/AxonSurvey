


## New Sampling Strategy: Representative Sampling

The goal is to develop a new sampling strategy, which I'll call **"representative sampling."** Here’s the core idea and steps:

### 1. Feature Extraction

- Load the images.
- Use one of our segmentation models to extract features from the different areas of each image.
- Specifically, run segmentation to determine the **axon density** in the image.

### 2. Analyzing Regions of Interest

- For each region of interest, analyze the distribution of axon density.

### 3. Sample Selection

- Select a sample of patches within each region of interest where:
    - The **average axon density** of the sample matches the axon density of the entire region (i.e., the population).
    - The samples include a **diverse variety** of axon densities—meaning, select patches that showcase a wide range of axon density values.

### 4. Goals of the Strategy

- The purpose is to ensure the selected sample:
    - **Accurately represents** the axon density of the population in each region of interest.
    - Includes a **wide variety** of axon densities to maintain diversity among samples.
    - Is implemented for each region of interest separately.

### 5. Rationale

- This strategy ensures that the sampling:
    - Provides a **solid estimation** of the axon density for the entire population.
    - Remains robust by including a **good variety** of images (in terms of axon density).
    - Supports reliable and representative data analysis.

---

**Summary:**  
The **representative sampling** strategy is designed to provide an accurate and robust estimate of axon density by selecting diverse samples that reflect the true distribution of axon densities for each region of interest. The goal is to build a sampling process that is both representative and solid, supporting high-quality analysis of the population.



