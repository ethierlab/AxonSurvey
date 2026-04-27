# ⚙️ Configuration Guide for AxonSurvey GUI

Welcome! Before you start using the AxonSurvey GUI for a new project, you need to make sure the application knows exactly where to find your data, images, and AI models. 

All of these settings are stored in a simple text file called **`config.json`** located inside the `gui` folder.

---

## 🔍 Key Settings to Verify

Open `gui/config.json` in any text editor (like Notepad, TextEdit, or Cursor). Here are the most important settings you need to check when starting a new project:

### 1. Your Image Data
* **`DATA_DIR`**: This is the folder where your raw rat brain scans are stored. 
  * *Default:* `"./data/project_scans"`
  * *What to check:* Make sure this folder actually exists and contains your images organized by rat and region.

### 2. Your Manual Tracings (Ground Truth)
* **`TRAINING_DIR`**: The folder containing the manual tracings used to train the model.
* **`TESTING_DIR`**: The folder containing the manual tracings used to test the model.
  * *Defaults:* `"./data/manual_tracings/2025-05-31_rat301_train"` and `..._test`
  * *What to check:* You **must** update these paths to match the dates and names of the tracing folders for your current project.

### 3. The AI Model
* **`USED_SEGMENTATION_MODEL_PATH`**: The exact file path to the trained AI model (usually a `.pth` file).
  * *Default:* `"./data/trained_models/default_model.pth"`
  * *What to check:* Ensure the `.pth` file actually exists at this location. If you trained a new model, update this path to point to your new file.

---

## 🚨 Common Problems & Solutions

### Problem: "Data directory not found" or "Cache directory not found"
**What it means:** The GUI cannot find the folders specified in `DATA_DIR` or `CACHE_DIR`.
**How to fix:** 
1. Open `gui/config.json`.
2. Check the spelling of your folders.
3. Make sure the folders have actually been created on your computer.

### Problem: "Segmentation model not found" when trying to Quick Trace
**What it means:** The app is trying to run the AI, but the model file is missing.
**How to fix:** 
1. Verify that your `.pth` model file is inside the `data/trained_models` folder.
2. Check that `USED_SEGMENTATION_MODEL_PATH` in `config.json` exactly matches the name of your model file.

### Problem: Windows File Paths
If you are copying and pasting folder paths on Windows (e.g., `C:\Users\Name\Desktop\data`), the backslashes (`\`) can break the JSON file.
**How to fix:** 
Always use **forward slashes** (`/`) or **double backslashes** (`\\`) in your `config.json`.
* ❌ *Bad:* `"C:\Users\Name\Desktop\data"`
* ✅ *Good:* `"C:/Users/Name/Desktop/data"`
* ✅ *Good:* `"C:\\Users\\Name\\Desktop\\data"`

---

## 💡 Tip: Relative vs Absolute Paths
* **Relative paths** start with `./` (which means "start from the main AxonSurvey folder"). Example: `"./data/project_scans"`
* **Absolute paths** spell out the entire location on your computer. Example: `"C:/Users/YourName/Documents/AxonSurvey/data/project_scans"`

If relative paths are confusing or not working, you can always replace them with the full absolute path to your folders!