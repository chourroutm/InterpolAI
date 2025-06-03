# InterpolAI

We recommend using a conda environment over a virtual environment.
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

   - **For Windows**
     ```bash
     conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
     ```
     
     ```bash
     pip install -r requirements.txt
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
## Activate the environment:
```bash
conda activate InterpolAI
```

## Usage
In the interpolation folder, you can find individual executable jupyter notebooks as listed:
1. `interpolAI_auto.ipynb` : searches through a folder of images and looking at the images missing in the folder, it will generates the missing images.
- **OR**
```bash
python main.py --mode auto --tile_size 1024 1024 --pth "\\10.99.68.178\Saurabh\manuscript_figs\data\HE_roi1\authentic\test"
```
2. `interpolAI_no_skip.ipynb`: on the destination folder, it will generates a given number of images between input images of the entire stack. (skip=2 means it will generate 2 images between every image pairs)
- **OR**
```bash
python main.py --mode no_skip --tile_size 1024 1024 --pth "\\10.99.68.178\Saurabh\manuscript_figs\data\HE_roi1\authentic\test" --skip 1 3 5 
```
3. `interpolAI_skip_haralick.ipynb`: on the destination folder, it will delete one in every image and generates the respective images between input images of the entire stack.
- **OR**
```bash
python main.py --mode skip --tile_size 1024 1024 --pth "\\10.99.68.178\Saurabh\manuscript_figs\data\HE_roi1\authentic\test" --skip 1
```