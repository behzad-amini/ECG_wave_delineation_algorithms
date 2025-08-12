# ECG Wave Delineation Algorithms Overview

## Introduction

This document serves as a comprehensive guide to various open-source ECG wave delineation algorithms and models. ECG wave delineation involves identifying the boundaries (onsets, offsets, and peaks) of key components in electrocardiogram (ECG) signals, such as P-waves, QRS complexes, and T-waves. These algorithms can be integrated into your project's system, where data is sent to multiple algorithms running in separate Docker containers. The system can then aggregate results (e.g., via voting, combining, or decision logic) for robust analysis.

For each algorithm, we cover:
- **Description**: Overview and functionality.
- **Strengths and Weaknesses**: Key advantages, limitations, and potential biases.
- **Input Format**: Required data structure for ECG signals.
- **Output Format**: Structure of the delineation results.
- **Installation and Dependencies**: Setup requirements.
- **Fine-Tuning and Improvements**: Options for customization or enhancement.
- **Docker Containerization Notes**: Guidance for building a Docker image, including sample Dockerfile snippets where applicable.
- **Additional Notes**: Any other relevant details, such as evaluation datasets or citations.

Algorithms are categorized into **Algorithmic (Non-Deep Learning)** for efficient, rule-based methods and **Deep Learning-Based** for more advanced, data-driven approaches. All information is based on repository READMEs and associated documentation as of August 11, 2025.

**Usage in Project**: Each Docker container can expose an API (e.g., via Flask or FastAPI for Python-based ones) to receive ECG data, process it, and return delineations. Your central system can orchestrate calls and fuse outputs.

## Algorithmic (Non-Deep Learning) Models

### 1. NeuroKit2
#### Description
NeuroKit2 is a Python toolbox for neurophysiological signal processing, including ECG delineation using methods like discrete wavelet transform (DWT). It identifies QRS complexes, P-peaks, T-peaks, and their onsets/offsets.

#### Strengths and Weaknesses
- **Strengths**: User-friendly with minimal code; supports signal simulation and visualization; versatile for various biosignals.
- **Weaknesses**: May underperform on noisy or atypical signals; EGG-related features are under development, indicating potential gaps in maturity for some analyses.

#### Input Format
- ECG signal as a NumPy array or Pandas Series.
- Required: Sampling rate (e.g., 3000 Hz).
- Example: `ecg_signal = nk.data("ecg_3000hz")`; supports single-lead signals.

#### Output Format
- Dictionary with:
  - `signals`: Processed ECG data.
  - `info`: Delineated waves (e.g., lists of indices for P-onsets, P-peaks, QRS-onsets, etc.).
- Example: `waves = {'ECG_P_Onsets': [indices], 'ECG_T_Peaks': [indices], ...}`.

#### Installation and Dependencies
- Install: `pip install neurokit2` or `conda install -c conda-forge neurokit2`.
- Dependencies: NumPy, Pandas, SciPy, Matplotlib (automatically handled by pip).

#### Fine-Tuning and Improvements
- Switch methods (e.g., `method="dwt"` vs. others) for delineation.
- Contribute new features via GitHub; open for user enhancements like custom preprocessing.

#### Docker Containerization Notes
- Use a Python base image.
- Sample Dockerfile:

```
FROM python:3.12-slim
RUN pip install neurokit2
WORKDIR /app
COPY app.py /app  # Your script to process ECG data
CMD ["python", "app.py"]
```

- Expose a port for API if needed (e.g., add Flask).

#### Additional Notes
- Evaluated on standard datasets; see examples at https://neuropsychology.github.io/NeuroKit/examples/ecg_delineate/ecg_delineate.html.

### 2. ECGdeli (MATLAB)
#### Description
ECGdeli is a MATLAB toolbox for ECG filtering and delineation, annotating onsets, peaks, and offsets of P-waves, QRS complexes, and T-waves in single or multi-lead ECGs.

#### Strengths and Weaknesses
- **Strengths**: Comprehensive filtering (baseline wander, frequency, isoline correction); supports multi-lead synchronization.
- **Weaknesses**: Provided "as is" with no performance guarantees; MATLAB dependency limits portability.

#### Input Format
- ECG as a matrix: Rows = time samples, columns = leads (standing vectors for single-lead).
- Example: Load from PhysioNet databases; processed via `Annotate_ECG_Multi.m`.

#### Output Format
- Fiducial Point Table (FPT): Table with timestamps (indices) for onsets/peaks/offsets per wave and lead.

#### Installation and Dependencies
- Clone repo; requires MATLAB with: Image Processing, Signal Processing, Statistics, Wavelet Toolboxes.
- No pip/conda; MATLAB-specific.

#### Fine-Tuning and Improvements
- Adjust filtering parameters in scripts; contribute via pull requests for algorithm enhancements.

#### Docker Containerization Notes
- Challenging due to MATLAB licensing; use MATLAB Compiler Runtime (MCR) for deployment.
- Sample: Build with MATLAB Docker image (official from MathWorks), but requires license. Alternative: Convert to Python if possible, or skip for open-source projects.
- Dockerfile example (assuming MCR installed):

```
FROM mathworks/matlab:r2025a  # Requires MathWorks account/license
COPY . /app
CMD ["matlab", "-batch", "Annotate_ExampleECG"]
```


#### Additional Notes
- Citation: Pilia et al. (2021), SoftwareX. Evaluated on PTB Diagnostic ECG Database.

### 3. pyECGdeli
#### Description
Python port of ECGdeli, focusing on QRS and R-peak detection with multi-lead support for P, QRS, T wave delineation.

#### Strengths and Weaknesses
- **Strengths**: Multi-lead detection without extra functions; zero-based indexing in FPT.
- **Weaknesses**: Work in progress (some algorithms missing); no performance guarantees.

#### Input Format
- ECG as NumPy array: Rows = time samples, columns = leads.
- Supports single or multi-lead.

#### Output Format
- FPT table with timestamps (indices) for waves; similar to ECGdeli but Python-friendly.

#### Installation and Dependencies
- Clone repo; dependencies not specified (likely NumPy, SciPy for signal processing).

#### Fine-Tuning and Improvements
- Ongoing development; add missing algorithms or tune parameters via code modifications.

#### Docker Containerization Notes
- Simple Python setup.
- Sample Dockerfile:

```
FROM python:3.12-slim
RUN pip install numpy scipy  # Assume these
COPY . /app
CMD ["python", "your_script.py"]
```


#### Additional Notes
- Evaluated on QT database (PhysioNet); errors: mean ± std = -2.00 ± 3.85 ms.

### 4. WTdelineator
#### Description
Wavelet-based ECG delineator in Python, based on Martínez et al. (2004), for P, QRS, T wave detection.

#### Strengths and Weaknesses
- **Strengths**: Established wavelet method; evaluated on standard databases.
- **Weaknesses**: Limited details; depends on specific databases like STAFFIII.

#### Input Format
- ECG signals from WFDB format (PhysioNet); e.g., single signal or database records.

#### Output Format
- Annotations (indices) for waves; details in output files (e.g., CSV).

#### Installation and Dependencies
- Requires WFDB-Python: `pip install wfdb`.
- Clone repo.

#### Fine-Tuning and Improvements
- Modify wavelet parameters in code; no explicit fine-tuning docs.

#### Docker Containerization Notes
- Python-based.
- Sample Dockerfile:

```
FROM python:3.12-slim
RUN pip install numpy scipy  # Assume these
COPY . /app
CMD ["python", "your_script.py"]
```

#### Additional Notes
- Evaluated on QT database (PhysioNet); errors: mean ± std = -2.00 ± 3.85 ms.

### 4. WTdelineator
#### Description
Wavelet-based ECG delineator in Python, based on Martínez et al. (2004), for P, QRS, T wave detection.

#### Strengths and Weaknesses
- **Strengths**: Established wavelet method; evaluated on standard databases.
- **Weaknesses**: Limited details; depends on specific databases like STAFFIII.

#### Input Format
- ECG signals from WFDB format (PhysioNet); e.g., single signal or database records.

#### Output Format
- Annotations (indices) for waves; details in output files (e.g., CSV).

#### Installation and Dependencies
- Requires WFDB-Python: `pip install wfdb`.
- Clone repo.

#### Fine-Tuning and Improvements
- Modify wavelet parameters in code; no explicit fine-tuning docs.

#### Docker Containerization Notes
- Python-based.
- Sample Dockerfile:
```
FROM python:3.12-slim
RUN pip install wfdb
COPY . /app
CMD ["python", "delineateSignal.py"]
```

#### Additional Notes
- Uses STAFFIII database for examples.

### 5. Prominence-Delineator
#### Description
Physiology-informed algorithm using peak prominence for detecting positions, onsets, offsets of P, R, T waves in single/multi-lead ECGs.

#### Strengths and Weaknesses
- **Strengths**: High F1-scores on datasets; interpretable; customizable parameters.
- **Weaknesses**: Constrained by physiological boundaries; potential for better prominence computation.

#### Input Format
- ECG as NumPy array or list: Single-lead (1D array) or multi-lead (list of arrays).
- Required: Sampling frequency.

#### Output Format
- Dictionary: Keys = wave types (e.g., 'P', 'R', 'T'); values = lists of positions/onsets/offsets.

#### Installation and Dependencies
- `pip install prominence-delineator`.
- Dependencies: Standard (NumPy, SciPy implied).

#### Fine-Tuning and Improvements
- Customize physiological parameters; develop new prominence methods.

#### Docker Containerization Notes
- Straightforward.
- Sample Dockerfile:
```
FROM python:3.12-slim
RUN pip install prominence-delineator
COPY app.py /app
CMD ["python", "/app/app.py"]
```


#### Additional Notes
- Example notebook available.

## Deep Learning-Based Models

### 6. DelineatorSwitchAndCompose
#### Description
Uses U-Net and W-Net architectures with synthetic data augmentation for delineating P, QRS, T waves. High F1-scores on QTDB/LUDB.

#### Strengths and Weaknesses
- **Strengths**: Handles arrhythmias; low errors (e.g., 5.8 ms); uses custom losses (BoundaryLoss, F1InstanceLoss).
- **Weaknesses**: Requires training data; computational intensive.

#### Input Format
- ECG signals as tensors; preprocessed segments.

#### Output Format
- Segmentation masks or boundaries for waves.

#### Installation and Dependencies
- PyTorch-based; clone repo, install requirements (not detailed, but typical: torch, numpy).

#### Fine-Tuning and Improvements
- Retrain with new data; adjust architectures or losses.

#### Docker Containerization Notes
- Use PyTorch Docker base.
- Sample: 
```
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "train.py"]
```


#### Additional Notes
- Paper: Jimenez-Perez et al. (arXiv); synthetic augmentation from segment pools.

### 7. ECG Deep Segmentation
#### Description
PyTorch models (ECG-SegNet, Cnn-SegNet) for segmenting ECG into P, QRS, T, extrasystoles, background.

#### Strengths and Weaknesses
- **Strengths**: Multiple preprocessing (differencing, FSST, BWR); high accuracy (up to 0.942); LSTM/CNN options.
- **Weaknesses**: Fixed window size (220); small datasets; extrasystole recall issues.

#### Input Format
- Matrix: N (samples) x T (220) x F (features, e.g., 1-2).

#### Output Format
- T x C tensor: Probabilities per class (Neutral, P, QRS, T, Extrasystole).

#### Installation and Dependencies
- Python 3.8; `pip install -r requirements.txt` (PyTorch, etc.).

#### Fine-Tuning and Improvements
- Alter network params; try new preprocessing; adjust window size (200-400).

#### Docker Containerization Notes
- GPU support if needed.
- Sample Dockerfile:

```
FROM python:3.8-slim
RUN pip install torch torchvision
COPY . /app
CMD ["python", "train.py"]
```


#### Additional Notes
- Datasets not public; uses Adam optimizer, CE loss.

### 8. U-Net-like ECG Segmentation (ckjoung/ecg-segmentation)
#### Description
Modified 1D U-Net with skip connections for delineating P, QRS, T in arrhythmias; optional classification branch.

#### Strengths and Weaknesses
- **Strengths**: >97% F1 on LUDB/QTDB; handles AFIB/VT.
- **Weaknesses**: Limited repo details; assumes standard setups.

#### Input Format
- 1D ECG signals; preprocessed for U-Net input.

#### Output Format
- Onsets/offsets as indices.

#### Installation and Dependencies
- PyTorch; training scripts in repo.

#### Fine-Tuning and Improvements
- Retrain on new data; modify architecture.

#### Docker Containerization Notes
- Similar to other PyTorch repos.
- Sample: Use pytorch base image.

#### Additional Notes
- PLOS One paper; code for training/inference.

### 9. torch_ecg
#### Description
PyTorch library with models (CRNN, U-Net, RR-LSTM) for ECG tasks, including wave delineation via sequence tagging.

#### Strengths and Weaknesses
- **Strengths**: Flexible configs; metrics like WaveDelineationMetrics; database integration.
- **Weaknesses**: Some benchmarks outdated; ongoing additions (U-Net++ planned).

#### Input Format
- Tensors: e.g., (batch, leads, length) like (2, 12, 4000).
- Preprocessors for numpy arrays.

#### Output Format
- WaveDelineationOutput: Dicts with fields for metrics (e.g., boundaries).

#### Installation and Dependencies
- `pip install torch-ecg`.
- PyTorch, NumPy.

#### Fine-Tuning and Improvements
- Custom configs (e.g., CNN backbones); add new models.

#### Docker Containerization Notes
- PyTorch-friendly.
- Sample:

```
FROM pytorch/pytorch:latest
RUN pip install torch-ecg
COPY app.py /app
CMD ["python", "app.py"]
```

#### Additional Notes
- Benchmarks for CPSC2021; open for contributions.

## Conclusion
This overview enables Dockerizing each algorithm for your system. Start with Python-based
