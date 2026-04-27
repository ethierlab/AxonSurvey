
## Dataset Page: Verification Checks

### Goal
Verify distribution balance across rats, regions, and slices.

### 1) Rat Distribution
- Check: The number of images per rat should be proportional to the number of regions in the underlying data for that rat.
- Categories:
  - 100% proportional
  - roughly proportional
  - unbalanced

### 2) Region Distribution
- Check: Equal distribution across regions (images per region should be equal).
- Categories:
  - equal
  - roughly equal
  - unbalanced

### 3) Slice Distribution
- Check: Equal distribution across slices (images per slice should be equal).
- Categories:
  - equal
  - roughly equal
  - unbalanced

### Category Definitions and Tolerances
- Equal: exact equality.
- Roughly equal / roughly proportional: approximately ±20%.
- Unbalanced: outside that range.

That’s going to work.


## Axon Survey GUI — New Dataset Statistics Page

### Goal
Add a new page to the Axon Survey GUI that displays dataset statistics. For now, assume there is only one dataset, though this may change in the future.

### Data Source and Ingestion
- Primary source: info.txt files
- Ingest each info.txt and parse into a specific data class that stores:
  - rat
  - bregma
  - region
  - possibly other fields (TBD)
  
### Statistics to Display
- Rat distribution
- Region distribution (independent of any rat ID)
- Bregma distribution:
  - Consider each slice as dependent on the rat
  - Example: rat 1 slice 1 is not the same as rat 2 slice 1, even if the slice number matches
  - Use all unique (rat, bregma) combinations
  - This is primarily to check if all of them are the same across the dataset (consistency check)

### GUI/UX Structure
- Display the statistics on a dedicated page
- Sections:
  - First two data sections are not collapsible
  - The third section may contain a lot of data, so it should be collapsible
    - Collapsed by default
    - Expandable via an arrow, similar to the collapsible menus on the Results page

### Verifications and Checks
- Provide verifications above each data section
- For rats:
  - Look at all rat folders in the project
  - Flag cases where some rat folders have fewer images than others
  - Note: This is tricky because it depends on the number of bregma slices; a more complex approach is needed

### Beyond info.txt — Additional Dataset Analysis
- We likely need to compute statistics from more than just info.txt
  - Incorporate information about the source data (the big images themselves), not just the sampled dataset
- Potential approach (to be determined):
  - Traverse the dataset and inspect the images to identify:
    - Possible regions
    - Regions actually present
  - Use this to build a more accurate picture of the dataset for verification

### Open Questions / To Be Determined
- Exact set of additional fields to store in the data class (beyond rat, bregma, region)
- Whether to scan the source images directly to infer possible regions and other metadata
- Final definition of the verification logic, especially for:
  - Handling varying counts due to differences in bregma slices
  - Consistency checks across unique (rat, bregma) combinations
- For the rats, we have to look at...



## Dataset validation page — tracing checker features

- Add the tracing checker features to the dataset validation page.
- The **most important** check is verifying that all tracing files are present.
- If any files are missing, this should appear **at the top** of the page.
- Actually, all of this information should be **at the top**.