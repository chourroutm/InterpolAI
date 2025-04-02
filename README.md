# InterpolAI

## Installation Steps

1. **Create a Conda Environment with Python 3.9**
   ```bash
   conda create --name InterpolAI python=3.9
   ```

2. **Activate the Conda Environment**
   ```bash
   conda activate InterpolAI
   ```

3. **Install Required Packages**
   - **For macOS (M1/M2/M4 Pro Chip)**
     ```bash
     pip install -r requirements_macos.txt
     ```
     This command installs the necessary packages optimized for macOS, including TensorFlow for Apple Silicon.

   - **For Other Systems**
     ```bash
     pip install -r requirements.txt
     ```

4. **Run the Application**
   ```bash
   python main.py
   ```

## Alternative Installation Method
If you prefer using yml files for installation on windows machines, you can use the following commands:
```bash
conda env create -f environment_3090.yml
```
or 
```bash
conda env create -f environment_4090.yml
```
This will create a conda environment with the necessary dependencies for running the application on NVIDIA GPUs. The `environment_3090.yml` is optimized for RTX 3090, while `environment_4090.yml` is optimized for RTX 4090.

**Activate the environment**
```bash
conda activate InterpolAI
```
**MACOS machines**, you can use the following commands:
```bash
conda env create -f environment_macos.yml
```
Activate the environment:
```bash
conda activate InterpolAI
```

## Usage
On the interpolation folder, you can find the:
1. `interpolAI_auto.ipynb`: searches through a folder of images and looking at the images missing in the folder, it will generates the missing images.
2. `interpolAI_no_skip.ipynb`: on the destination folder, it will generates a given number of images between input images of the entire stack.
3. `FILM_Final_validation_backup.ipynb`: on the destination folder, it will delete one in every image and generates the respective images between input images of the entire stack.
