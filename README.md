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
## Model/weights download:
Please download the model folder from the following Google Drive link: [model](https://drive.google.com/drive/folders/16a4zhopq8AfKCADXxBwuYccGr_PnBRlt?usp=sharing)  
Once downloaded please place the model folder inside  the interpolation directory of the InterpolAI repository. 
## Usage
In the interpolation folder, you can find individual executable jupyter notebooks as listed:
1. `interpolAI_auto.ipynb` : loads and reads through a folder of images and detects missing images using filenames in the folder, algorithm will then generate the missing images.
- **OR** run the following main.py with mode auto
```bash
python main.py --mode auto --tile_size 1024 1024 --pth "\\10.99.68.178\Saurabh\manuscript_figs\data\HE_roi1\authentic\test"
```
2. `interpolAI_no_skip.ipynb`: loads and reads through a folder of images, algorithm will then generate a given number of images, as defined by users the skip flag, between each pair of images in the input folder. 
- **OR** run the following main.py with mode no_skip
```bash
python main.py --mode no_skip --tile_size 1024 1024 --pth "\\10.99.68.178\Saurabh\manuscript_figs\data\HE_roi1\authentic\test" --skip 1 3 5 
```
3. `interpolAI_skip_haralick.ipynb`: loads and reads through a folder of images, algorithm will then skip images in the folder and generate the skipped images, as defined by the users skip flag. 
- **OR** run the following main.py with mode skip
```bash
python main.py --mode skip --tile_size 1024 1024 --pth "\\10.99.68.178\Saurabh\manuscript_figs\data\HE_roi1\authentic\test" --skip 1
```