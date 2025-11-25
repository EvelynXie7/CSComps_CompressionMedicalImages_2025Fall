# Region of Interest Medical Image Compression with JPEG, JPEG2000, and SPIHT

This repository implements and compares three medical image compression algorithms—**JPEG**, **JPEG2000**, and **SPIHT(Set Partitioning in Hierarchical Trees)**—with a focus on **region-of-interest (ROI)** preservation. The goal is to evaluate how well each method compresses CT and MRI scans while maintaining diagnostic quality in clinically important regions such as tumors.

This project was developed for Carleton College Computer Science Comps 2025–2026.

---

## Overview

This repository provides:

- JPEG
- JPEG2000 with ROI
- SPIHT with ROI
- Pipelines for the evaluation of three algorithms on CT and MRI scans

---

## Datasets

This project uses two publicly available, annotated medical imaging datasets: BraTS 2021 and KiTS19.

### **KiTS19: Kidney Tumor CT**

https://github.com/neheller/kits19
Contains CT volumes with kidney and tumor segmentations.

### **BraTS 2021: Brain Tumor MRI**

https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/
Contains multi-modal MRI scans with detailed tumor segmentations.

---

## Features

### **JPEG Compression (Baseline)**

File:

- `jpeg.py`

Capabilities:

- Standard DCT, quantization, and entropy encoding
- Baseline (non-ROI) compression and decompression

---

### **SPIHT (Set Partitioning in Hierarchical Trees)**
Files:

- `SPIHT_encoder.py`
- `SPIHT_decoder.py`
- `spiht_roi_pipeline.py`
  
Capabilities:

- Standard SPIHT encoder/decoder
- ROI-based SPIHT compression and decompression

---

### **JPEG2000 with ROI**

File:

- `jpeg2000/jp2_glymur.py`

Capabilities:

- JPEG2000 encoding/decoding using the `glymur` library  
- ROI-based JPEG2000 compression and decompression

---

## Installation

### 1. Install Required Python Packages

```bash
pip install -r requirements.txt
```

### 2. Install Glymur and OpenJPEG (Required for JPEG2000)

Install glymur:

```bash
pip install glymur
```

Install OpenJPEG:

**macOS (Homebrew):**

```bash
brew install openjpeg
```

**Ubuntu/Debian:**

```bash
sudo apt-get install libopenjp2-7
```

**Conda (any OS):**

```bash
conda install -c conda-forge openjpeg
```

---
## Authors

- Rui Shen

- Justin Vaughn

- Lucie Wolf

- Evelyn Xie

Advised by Layla Oesper
