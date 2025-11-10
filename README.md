# Super-Resolution for Soil Moisture: Dataset and Baseline Models
This repository contains the code and links to dataset and models used for Soil moisture Super Resolution Study over California's Central Valley.

## Installation and Setup
To run the models and scripts in this repository, ensure your system meets the following requirements:

### Prerequisites
- Python 3.8 or higher
- conda
- Training Dataset

1. **Clone the Repository**
   ```bash
   git clone https://github.com/LUMS-WIT/WIT-SMSR-Cv.git
   cd WIT-SMSR-Cv

2. **Install Dependencies using conda**
   ```bash
   conda env create -f requirements.yml
   conda activate geo

## Usage

The dataset would be made available on publishing of the study. You need to download the dataset to execute the training and test scripts.


### `Train`

  - Command:
    ```bash
    python main.py
    ```

### `Test`

  - Command:
    ```python
    ```

### `Inference`

  - Command:
    ```python
    ```

## Results

### Quantitative Assessments  
*Upper-bound results are shown in parentheses and top performance in bold.*

| Method | Upscaling Factor | ubRMSE | MSE | Bias | PSNR (dB) | Pearson R |
|:--------|:----------------:|:-------:|:----:|:----:|:----------:|:----------:|
| **Baseline** | 3× | 0.22 (0.0983) | 0.0893 (0.0176) | 0.1126 (0.0148) | 11.21 (18.27) | 0.3973 (0.8865) |
| **SRCNN** |  | 0.1354 (0.0424) | 0.0275 (0.0029) | -0.0244 (-0.0011) | 16.15 (26.45) | 0.5797 (0.9744) |
| **Transformer** |  | 0.0849 (0.0264) | 0.0163 (0.0013) | -0.0139 (0.0056) | 20.51 (30.34) | 0.8847 (0.9905) |
| **TransformerSkip** |  | 0.0816 (0.0260) | 0.0145 (0.0012) | **-0.0183** (0.0044) | 20.82 (30.53) | 0.8920 (0.9908) |
| **GeoTransformerSR** |  | **0.0692** (0.0265) | **0.0100** (0.0014) | -0.0190 (0.0024) | **21.80** (30.06) | **0.9263** (0.9905) |
| **Baseline** | 9× | 0.2152 (0.1186) | 0.0947 (0.0305) | 0.1103 (0.0133) | 10.95 (16.32) | 0.3629 (0.8319) |
| **SRCNN** |  | 0.1320 (0.0766) | 0.0324 (0.0113) | -0.0429 (-0.0051) | 15.33 (20.44) | 0.3218 (0.8668) |
| **Transformer** |  | 0.0753 (0.0342) | 0.0139 (0.0026) | -0.0171 (0.0032) | 20.95 (27.43) | 0.8590 (0.9753) |
| **TransformerSkip** |  | 0.0712 (0.0307) | 0.0130 (0.0023) | -0.0152 (0.0009) | 21.47 (28.30) | 0.8703 (0.9806) |
| **GeoTransformerSR** |  | **0.0578** (0.0312) | **0.0080** (0.0020) | **-0.0111** (-0.0061) | **23.09** (28.37) | **0.9126** (0.9797) |

### Qualitative Assessments

![Results](images\fig_2.png)

## Citation
If you use this project in your research, please cite:
```bibtex



